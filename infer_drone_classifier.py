"""
infer_drone_classifier.py
==========================
Run the trained drone-vs-non-drone patch classifier on a video.

For every frame t, it looks GAP frames back, computes the two-frame
difference:

        diff = |frame[t] - frame[t - GAP]|

(exactly the same signal the model was trained on), feeds it through the
model, and overlays the predicted "Drone probability" percentage directly
on the output video.

USAGE
-----
    python infer_drone_classifier.py --video path/to/clip.mp4 --weights results.pth

Optional:
    --output out.mp4       Where to save the annotated video (default: <video>_annotated.mp4)
    --gap 5                Must match the gap used during training (default 5)
    --threshold 0.5         Probability above which a frame is called "DRONE"
    --display               Show a live preview window while processing
    --no_save                Skip writing an output video (useful with --display only)
"""

import os
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# =====================================================================
# Config — MUST match the values used in train_drone_classifier.py
# =====================================================================
IMG_W = 640
IMG_H = 480
PATCH_GRID = 4
PATCH_W = IMG_W // PATCH_GRID
PATCH_H = IMG_H // PATCH_GRID
PAD = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.ToTensor(),
])


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
# Model summary printer (layers, conv details, param counts)
# =====================================================================
def print_model_summary(model: nn.Module):
    """Print every leaf layer, Conv2d-specific details, and parameter counts.

    No forward hooks / dummy input here since PatchClassifier6.forward()
    expects a 5D (B, NP, C, H, W) patch tensor rather than a plain image
    tensor, so output shapes are skipped to keep this safe to call right
    after the model is constructed or loaded.
    """
    print("=" * 80)
    print(f"MODEL: {model.__class__.__name__}")
    print("=" * 80)

    total_params = 0
    trainable_params = 0
    conv_layers = []
    leaf_count = 0

    print(f"\n{'Layer':<35}{'Type':<20}{'Params'}")
    print("-" * 80)

    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # skip containers, only leaf layers
        leaf_count += 1

        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        trainable_params += trainable

        print(f"{name:<35}{module.__class__.__name__:<20}{params}")

        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    print("-" * 80)
    print(f"\nTotal leaf layers: {leaf_count}")
    print(f"Total Conv2d layers: {len(conv_layers)}")

    if conv_layers:
        print("\nConv2d Layer Details:")
        print("-" * 80)
        for name, conv in conv_layers:
            print(f"  {name}:")
            print(f"    in_channels  = {conv.in_channels}")
            print(f"    out_channels = {conv.out_channels}")
            print(f"    kernel_size  = {conv.kernel_size}")
            print(f"    stride       = {conv.stride}")
            print(f"    padding      = {conv.padding}")
            print(f"    groups       = {conv.groups}")
            print(f"    bias         = {conv.bias is not None}")
            print(f"    params       = {sum(p.numel() for p in conv.parameters())}")

    print("\n" + "=" * 80)
    print(f"TOTAL PARAMETERS:       {total_params:,}")
    print(f"TRAINABLE PARAMETERS:   {trainable_params:,}")
    print(f"NON-TRAINABLE PARAMS:   {total_params - trainable_params:,}")
    print("=" * 80 + "\n")


# =====================================================================
# Helpers
# =====================================================================
def diff_to_patch_tensor(diff_gray, pad_fn):
    """Grayscale diff image [H, W] uint8 -> [1, 16, 1, PATCH_H+2, PATCH_W+2] tensor,
    matching exactly how training patches were built."""
    img_pil = Image.fromarray(diff_gray)
    img_tensor = val_transform(img_pil)  # [1, IMG_H, IMG_W]

    patches = []
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            x0 = col * PATCH_W
            y0 = row * PATCH_H
            patch = img_tensor[:, y0:y0 + PATCH_H, x0:x0 + PATCH_W]
            patches.append(pad_fn(patch))

    stacked = torch.stack(patches)          # [16, 1, PATCH_H+2, PATCH_W+2]
    return stacked.unsqueeze(0)              # [1, 16, 1, PATCH_H+2, PATCH_W+2]


def draw_overlay(frame_bgr, prob, threshold, buffering=False):
    """Draws a percentage readout + label banner on the frame (in place)."""
    h, w = frame_bgr.shape[:2]
    banner_h = 60
    cv2.rectangle(frame_bgr, (0, 0), (w, banner_h), (0, 0, 0), thickness=-1)

    if buffering:
        text = "Buffering..."
        color = (200, 200, 200)
    else:
        pct = prob * 100.0
        label = "DRONE" if prob >= threshold else "NON-DRONE"
        color = (0, 0, 255) if prob >= threshold else (0, 200, 0)  # BGR: red / green
        text = f"{label}  |  Drone probability: {pct:5.1f}%"

    cv2.putText(frame_bgr, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame_bgr


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--weights", type=str, default="results.pth", help="Path to trained weights")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated video")
    parser.add_argument("--gap", type=int, default=5, help="Frame gap for two-frame difference (must match training)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold to call it DRONE")
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    parser.add_argument("--no_save", action="store_true", help="Do not write an output video file")
    parser.add_argument("--summary", action="store_true", help="Print full model layer/parameter summary and exit setup checks")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print("=" * 70)
    print("DRONE vs NON-DRONE  |  VIDEO INFERENCE")
    print("=" * 70)
    print(f"Video      : {args.video}")
    print(f"Weights    : {args.weights}")
    print(f"Frame gap  : {args.gap}")
    print(f"Threshold  : {args.threshold}")
    print(f"Device     : {DEVICE}")

    # ---- Load model ----
    model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
    state_dict = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.\n")

    if args.summary:
        print_model_summary(model)

    pad_fn = nn.ReflectionPad2d(PAD)

    # ---- Open video ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source video: {src_w}x{src_h} @ {fps:.1f} fps, {total_frames} frames\n")

    # ---- Output writer ----
    writer = None
    if not args.no_save:
        out_path = args.output or (os.path.splitext(args.video)[0] + "_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w, src_h))
        print(f"Annotated video will be saved to: {out_path}\n")

    gray_buffer = deque(maxlen=args.gap + 1)
    frame_probs = []  # track per-frame drone probability for a final video-level summary

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing")
    frame_count = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_full, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        gray_buffer.append(gray_resized)

        if len(gray_buffer) == args.gap + 1:
            older, newer = gray_buffer[0], gray_buffer[-1]
            diff = cv2.absdiff(newer, older)

            with torch.no_grad():
                patch_tensor = diff_to_patch_tensor(diff, pad_fn).to(DEVICE)
                logit = model(patch_tensor)
                prob = torch.sigmoid(logit).item()

            frame_probs.append(prob)
            draw_overlay(frame_bgr, prob, args.threshold, buffering=False)
        else:
            # Not enough history yet to compute a gap-frame diff
            draw_overlay(frame_bgr, 0.0, args.threshold, buffering=True)

        if writer is not None:
            writer.write(frame_bgr)
        if args.display:
            cv2.imshow("Drone vs Non-Drone Inference", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Stopped early by user (pressed 'q').")
                break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # ---- Video-level summary ----
    print("\n" + "=" * 70)
    if len(frame_probs) == 0:
        print("No frames had enough history to run inference (video too short for --gap).")
    else:
        probs_np = np.array(frame_probs)
        avg_prob = probs_np.mean()
        max_prob = probs_np.max()
        pct_frames_drone = (probs_np >= args.threshold).mean() * 100.0

        print("INFERENCE SUMMARY")
        print(f"  Frames analyzed          : {len(frame_probs)}")
        print(f"  Average drone probability: {avg_prob * 100:.1f}%")
        print(f"  Peak drone probability   : {max_prob * 100:.1f}%")
        print(f"  Frames classified DRONE  : {pct_frames_drone:.1f}% of analyzed frames")
        verdict = "DRONE" if avg_prob >= args.threshold else "NON-DRONE"
        print(f"  Overall verdict          : {verdict}  (avg prob {avg_prob * 100:.1f}% "
              f"{'>=' if avg_prob >= args.threshold else '<'} threshold {args.threshold * 100:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()