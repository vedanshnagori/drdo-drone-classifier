"""
Grad-CAM for the patch-based drone classifier.

What this does:
  1. Loads your trained model.
  2. Loads a single frame (jpg) and cuts it into the same 16 patches
     used during training.
  3. Runs a forward + backward pass and captures which regions of each
     patch most influenced the "drone" prediction.
  4. Stitches the 16 patch-heatmaps back into one full-frame heatmap
     and overlays it on the original image so you can SEE where the
     model is looking.

Usage:
  1. Set BEST_MODEL_PATH, FRAME_PATH, OUTPUT_PATH below.
  2. Run: python grad_cam_drone.py
  3. Open the saved overlay image. Red/yellow = high influence on the
     prediction, blue = low influence.

What to look for:
  - On a frame that actually contains a drone: is the hot region ON
    the drone, or somewhere else in the background?
  - On a frame with NO drone that the model still flags as "drone":
    where is it looking? That tells you what shortcut it's using.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# =======================
# Must match your training script
# =======================
IMG_W = 640
IMG_H = 480
PATCH_GRID = 4
PATCH_W = IMG_W // PATCH_GRID
PATCH_H = IMG_H // PATCH_GRID
PAD = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# CHANGE THESE THREE PATHS
# =======================
BEST_MODEL_PATH = "results.pth"
FRAME_PATH = "frame1.jpg"
OUTPUT_PATH = "frameoutput.jpg"


# ==========================================================
# Model architecture (identical copy from training script,
# needed here so this script can run standalone)
# ==========================================================
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
        return F.linear(input, w, self.bias)


class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = PoTConv2d(in_ch, in_ch, 3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = PoTConv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x


class PatchClassifier6(nn.Module):
    def __init__(self, in_ch=1, base_filters=16, patch_feature_dim=32):
        super().__init__()
        self.stem = nn.Sequential(
            PoTConv2d(in_ch, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
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
        feats = self.forward_backbone(x)
        feats = feats.view(B, NP, -1)
        pooled = feats.max(dim=1)[0]
        return self.fc_final(pooled)


# ==========================================================
# Grad-CAM
# ==========================================================
class GradCAM:
    """
    Hooks into `target_layer` to grab its output (activations) during the
    forward pass and its gradient during the backward pass. Combining
    them tells us which spatial locations mattered most for the
    prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, patches):
        """
        patches: [1, 16, 1, H, W] tensor, requires_grad not needed on the
        input itself -- we only need gradients w.r.t. the hooked layer.
        Returns: cam [16, h, w] (values 0-1) and the predicted probability.
        """
        self.model.zero_grad()
        logits = self.model(patches)          # [1, 1]
        prob = torch.sigmoid(logits)
        logits.sum().backward()

        # activations / gradients shape: [16, C, h, w]  (16 patches, batch=1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # global-average-pool the gradient per channel
        cam = F.relu((weights * self.activations).sum(dim=1))     # weighted sum over channels
        cam = cam.cpu().numpy()

        # Normalize globally across all 16 patches so they stay comparable
        # to each other (rather than each patch stretching its own 0-1 range).
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam, prob.item()


def load_and_prepare_frame(frame_path):
    """Loads a raw frame and cuts it into the same 16 padded patches used in training."""
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img_pil = Image.fromarray(img)
    img_tensor = transforms.ToTensor()(img_pil)   # [1, IMG_H, IMG_W]

    pad_fn = torch.nn.ReflectionPad2d(PAD)
    patches = []
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            x0, y0 = col * PATCH_W, row * PATCH_H
            p = img_tensor[:, y0:y0 + PATCH_H, x0:x0 + PATCH_W]
            patches.append(pad_fn(p))
    patches = torch.stack(patches).unsqueeze(0)  # [1, 16, 1, ph, pw]
    return img, patches


def overlay_cam_on_frame(raw_frame, cam):
    """Stitches the 16 patch heatmaps back into one full-frame heatmap and overlays it."""
    grid_h = PATCH_H + 2 * PAD
    grid_w = PATCH_W + 2 * PAD
    full_heatmap = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    idx = 0
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            patch_cam = cam[idx]
            patch_cam_resized = cv2.resize(patch_cam, (grid_w, grid_h))
            # Drop the reflection-padding border before placing it back
            patch_cam_cropped = patch_cam_resized[PAD:PAD + PATCH_H, PAD:PAD + PATCH_W]
            y0, x0 = row * PATCH_H, col * PATCH_W
            full_heatmap[y0:y0 + PATCH_H, x0:x0 + PATCH_W] = patch_cam_cropped
            idx += 1

    heatmap_uint8 = np.uint8(255 * full_heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    raw_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(raw_bgr, 0.6, heatmap_color, 0.4, 0)
    return overlay


if __name__ == "__main__":
    model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Target the output of block2 -- the last conv block before pooling.
    # This is the finest-resolution feature map available, so the resulting
    # heatmap will localize as precisely as this architecture allows.
    gradcam = GradCAM(model, model.block2)

    raw_frame, patches = load_and_prepare_frame(FRAME_PATH)
    patches = patches.to(DEVICE)

    cam, prob = gradcam.generate(patches)
    overlay = overlay_cam_on_frame(raw_frame, cam)

    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Predicted drone probability: {prob:.4f}")
    print(f"Grad-CAM overlay saved to: {OUTPUT_PATH}")
