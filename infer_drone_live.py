"""
infer_drone_live.py
=====================
LIVE drone-vs-non-drone prediction on a video, using a fixed-size circular
frame buffer instead of a growing history.

THE BUFFER TRICK (what you asked for)
--------------------------------------
Instead of keeping a sliding window/deque of the last GAP frames (which
copies data around every step), we keep a small pre-allocated buffer of
exactly GAP slots:

    buffer = [None] * GAP

For frame index t:
    idx    = t % GAP                     # which slot this frame maps to
    older  = buffer[idx]                 # this IS frame[t - GAP] already
    diff   = |frame[t] - older|          # the model's input
    ... run prediction on diff ...
    buffer[idx] = frame[t]               # overwrite frame[t-GAP] in place

Because slot `idx` was last written exactly GAP steps ago (the buffer
cycles every GAP frames), whatever is sitting in `buffer[idx]` right before
we overwrite it IS frame[t - GAP] — no separate history list needed. Only
GAP frames (grayscale, resized) are ever resident in memory at once,
regardless of how long the video is.

USAGE
-----
    python infer_drone_live.py --video path/to/clip.mp4 --weights results.pth --display

    # Live-paced playback (waits to match the source FPS instead of running
    # as fast as possible) + also save the annotated video:
    python infer_drone_live.py --video clip.mp4 --weights results.pth --display --realtime --output out.mp4

    # Use a webcam instead of a file:
    python infer_drone_live.py --video 0 --weights results.pth --display --realtime
"""

import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =====================================================================
# Config — MUST match train_drone_classifier.py
# =====================================================================
IMG_W = 640
IMG_H = 480
PATCH_GRID = 4
PATCH_W = IMG_W // PATCH_GRID
PATCH_H = IMG_H // PATCH_GRID
PAD = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([transforms.ToTensor()])


# =====================================================================
# Model definition — identical to training script so state_dict loads cleanly
# =====================================================================
class PoTQuantizerSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_exp=-8, max_exp=7):
        sign = torch.sign(x)
        x_abs = torch.abs(x)
        eps = 2.0 ** (min_exp - 1)
        x_abs = torch.clamp(x_abs, min=eps)
        log2_x = torch.round(torch.log2(x_abs))
        log2_x = torch.clamp(log2_x, min=min_exp, max=max_exp)
        x_pot = sign * (2.0 ** log2_x)
        x_pot[torch.abs(x) < (2.0 ** min_exp)] = 0.0
        return x_pot

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class PoTConv2d(nn.Conv2d):
    def forward(self, input):
        w = PoTQuantizerSTE.apply(self.weight)
        return self._conv_forward(input, w, self.bias)


class PoTLinear(nn.Linear):
    def forward(self, input):
        w = PoTQuantizerSTE.apply(self.weight)
        return nn.functional.linear(input, w, self.bias)


class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = PoTConv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = PoTConv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.act(self.pw_bn(self.pointwise(x)))
        return x


class PatchClassifier6(nn.Module):
    def __init__(self, in_ch=1, base_filters=16, patch_feature_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            PoTConv2d(in_ch, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.block1 = DWConvBlock(base_filters, base_filters * 2, stride=2)
        self.block2 = DWConvBlock(base_filters * 2, base_filters * 4, stride=2)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_patch = PoTLinear(base_filters * 4, patch_feature_dim)
        self.fc_final = PoTLinear(patch_feature_dim, 1)

    def forward_backbone(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc_patch(x)

    def forward(self, patches):
        B, NP, C, H, W = patches.shape
        x = patches.view(B * NP, C, H, W)
        feats = self.forward_backbone(x).view(B, NP, -1)
        pooled = feats.max(dim=1)[0]
        return self.fc_final(pooled)


# =====================================================================
# Helpers
# =====================================================================
def diff_to_patch_tensor(diff_gray, pad_fn):
    """Grayscale diff [H, W] uint8 -> [1, 16, 1, PATCH_H+2, PATCH_W+2] tensor."""
    img_pil = Image.fromarray(diff_gray)
    img_tensor = val_transform(img_pil)  # [1, IMG_H, IMG_W]

    patches = []
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            x0 = col * PATCH_W
            y0 = row * PATCH_H
            patch = img_tensor[:, y0:y0 + PATCH_H, x0:x0 + PATCH_W]
            patches.append(pad_fn(patch))

    stacked = torch.stack(patches)
    return stacked.unsqueeze(0)


def draw_overlay(frame_bgr, prob, threshold, fps_display, buffering=False):
    h, w = frame_bgr.shape[:2]
    banner_h = 60
    cv2.rectangle(frame_bgr, (0, 0), (w, banner_h), (0, 0, 0), thickness=-1)

    if buffering:
        text = "Buffering (filling frame ring buffer)..."
        color = (200, 200, 200)
    else:
        pct = prob * 100.0
        label = "DRONE" if prob >= threshold else "NON-DRONE"
        color = (0, 0, 255) if prob >= threshold else (0, 200, 0)  # BGR
        text = f"{label}  |  Drone probability: {pct:5.1f}%"

    cv2.putText(frame_bgr, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"{fps_display:.1f} FPS", (w - 150, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                         help="Path to video file, or a webcam index like 0")
    parser.add_argument("--weights", type=str, default="results.pth", help="Path to trained weights")
    parser.add_argument("--gap", type=int, default=5, help="Frame gap (must match training)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for DRONE label")
    parser.add_argument("--display", action="store_true", help="Show a live window")
    parser.add_argument("--realtime", action="store_true",
                         help="Pace playback to match the source video's FPS (true 'live' viewing)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to also save the annotated video")
    args = parser.parse_args()

    # Allow "--video 0" to mean webcam index 0
    video_source = int(args.video) if args.video.isdigit() else args.video
    if isinstance(video_source, str) and not os.path.exists(video_source):
        raise FileNotFoundError(f"Video not found: {video_source}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print("=" * 70)
    print("DRONE vs NON-DRONE  |  LIVE INFERENCE (ring-buffer frame differencing)")
    print("=" * 70)
    print(f"Source     : {video_source}")
    print(f"Weights    : {args.weights}")
    print(f"Frame gap  : {args.gap}  (ring buffer holds exactly {args.gap} frames)")
    print(f"Threshold  : {args.threshold}")
    print(f"Device     : {DEVICE}")

    model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.\n")

    pad_fn = nn.ReflectionPad2d(PAD)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or IMG_W
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or IMG_H
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = 1.0 / fps if fps > 0 else 1.0 / 25.0
    print(f"Source: {src_w}x{src_h} @ {fps:.1f} fps\n")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (src_w, src_h))
        print(f"Also saving annotated video to: {args.output}\n")

    # --- THE FIXED-SIZE RING BUFFER ---
    # Only `gap` grayscale frames are ever held in memory. `ring[idx]` gets
    # overwritten every `gap` steps, which is exactly frame[t - gap] the
    # moment before we overwrite it.
    ring = [None] * args.gap

    t = 0
    last_time = time.time()
    fps_display = 0.0

    print("Starting live inference. Press 'q' in the display window to stop.\n")

    while True:
        loop_start = time.time()
        ok, frame_bgr = cap.read()
        if not ok:
            break

        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_full, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        idx = t % args.gap
        older = ring[idx]  # this is frame[t - gap], if the buffer has cycled once already

        if older is not None:
            diff = cv2.absdiff(gray_resized, older)
            with torch.no_grad():
                patch_tensor = diff_to_patch_tensor(diff, pad_fn).to(DEVICE)
                logit = model(patch_tensor)
                prob = torch.sigmoid(logit).item()
            buffering = False
        else:
            prob = 0.0
            buffering = True

        # Overwrite frame[t - gap] with the current frame — this is the
        # memory-saving step: no growing list, no deque, same gap-sized
        # buffer reused for the entire video.
        ring[idx] = gray_resized

        now = time.time()
        inst_fps = 1.0 / max(now - last_time, 1e-6)
        fps_display = 0.9 * fps_display + 0.1 * inst_fps if fps_display > 0 else inst_fps
        last_time = now

        draw_overlay(frame_bgr, prob, args.threshold, fps_display, buffering=buffering)

        if not buffering:
            pct = prob * 100.0
            label = "DRONE" if prob >= args.threshold else "NON-DRONE"
            print(f"  frame {t:6d} | drone probability: {pct:5.1f}% | {label}")

        if writer is not None:
            writer.write(frame_bgr)

        if args.display:
            cv2.imshow("Live Drone vs Non-Drone", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Stopped by user.")
                break

        if args.realtime:
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        t += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"Done. Processed {t} frames using a fixed {args.gap}-frame ring buffer.")
    print("=" * 70)


if __name__ == "__main__":
    main()
