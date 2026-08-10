"""
train_drone_classifier.py
==========================
Drone vs Non-Drone classifier trained on TWO-FRAME DIFFERENCE images.

INPUT LAYOUT (edit DRONE_DIR / NON_DRONE_DIR below, or pass --drone_dir /
--non_drone_dir on the command line):

    drones/
        clip_001.mp4
        clip_002.mp4
        ...
    non-drones/
        clip_101.mp4
        clip_102.mp4
        ...

PIPELINE
--------
1.  Every video is walked frame-by-frame (sequential read, no random seeks
    -> fast). For every frame t we compute:

            diff = |frame[t] - frame[t - GAP]|      (GAP = 5 by default)

    This is the "two-frame difference" trick: anything static (sky, trees,
    horizon, sensor noise) cancels out, anything that moved (a drone or a
    bird) shows up as a bright blob. The diff image (not the raw RGB frame)
    is what the network actually trains on.

2.  Smart filtering of low-signal diffs (this is the "skip if too dark"
    logic you asked for):

      - DRONE clips: if a diff frame has fewer than
        MIN_ACTIVE_PIXELS_DRONE pixels above ACTIVE_PIXEL_THRESH, the drone
        is essentially invisible in that frame (e.g. hovering perfectly
        still, or too small/far to register any motion). Training on a
        frame labeled "drone" that shows no drone-like signal teaches the
        network to associate the POSITIVE label with pure noise, which
        hurts precision. -> such frames are SKIPPED for the drone class.

      - NON-DRONE clips: a mostly-static diff is actually a *correct* and
        useful negative example (empty sky, swaying trees, etc. really do
        produce near-zero diffs), so we do NOT apply the same aggressive
        filter here. We only drop completely degenerate frames (max pixel
        difference below MIN_MAX_DIFF_NONDRONE), which almost always means
        a frozen/duplicated frame or a decoder glitch rather than a real
        negative example.

    This asymmetric filtering is the "smart decision" a working ML
    engineer would make: don't discard valid negatives, but don't let
    invisible positives poison the positive class either.

3.  Kept diff frames are cached to disk once (as .jpg) under
    ./diff_cache/drones/<video_name>/ and ./diff_cache/non-drones/<video_name>/
    so re-running the script doesn't require re-decoding every video.

4.  Videos (not individual frames) are split into train/val, stratified by
    class, so frames from the same clip never leak across the split.

5.  A lightweight power-of-two (PoT) quantization-aware patch classifier
    (same architecture family used for the Kria FPGA deployment target) is
    trained with a Binary Focal Loss, OneCycleLR, and a "hard PoT" weight
    swap during validation so the reported metrics reflect the ACTUAL
    quantized weights that will ship, not the FP32 shadow weights.

6.  Final weights are saved to RESULTS_PATH ("results.pth" by default, in
    the same folder as this script).
"""

import os
import glob
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# =====================================================================
# 1. CONFIG — edit these paths / knobs
# =====================================================================
# NOTE: argparse parsing, the derived path globals, and the startup banner
# used to live at module top-level. On Windows, DataLoader(num_workers>0)
# spawns worker processes that each re-IMPORT this file from scratch — so
# any code sitting at module top-level (outside a function or an
# `if __name__ == "__main__":` guard) runs again in every worker. That was
# printing the banner N times (once per worker) and piling up redundant
# work/handles that contributed to the Windows shared-memory crash. All of
# that setup now happens inside setup_config(), which is only called from
# inside the `if __name__ == "__main__":` guard at the bottom of the file,
# so workers never execute it.
IMG_W = 640
IMG_H = 480
PATCH_GRID = 4                       # 4x4 = 16 patches
PATCH_W = IMG_W // PATCH_GRID        # 160
PATCH_H = IMG_H // PATCH_GRID        # 120
PAD = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- motion / darkness filtering knobs ---
ACTIVE_PIXEL_THRESH = 20     # a pixel counts as "active" if |diff| > this
MIN_ACTIVE_PIXELS_DRONE = 8  # drone frames with fewer active px than this are SKIPPED
MIN_MAX_DIFF_NONDRONE = 3    # non-drone frames are only dropped if basically all-zero

MAX_FRAMES_PER_VIDEO = 3000  # safety cap so one huge video doesn't dominate
FRAME_STRIDE = 1             # set >1 to subsample (e.g. 2 = every other eligible frame)

# --- training hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
VAL_FRACTION = 0.15
RANDOM_SEED = 42

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")

# These are set by setup_config() (called only from the __main__ guard) but
# declared here so every function in the file can see them by name.
DRONE_DIR = None
NON_DRONE_DIR = None
CACHE_DIR = None
GAP = None
FORCE_REBUILD_CACHE = None
RESULTS_PATH = None


def setup_config():
    """Parses CLI args, sets the derived module-level config globals, seeds
    RNGs, and prints the startup banner. Only ever called once, from inside
    `if __name__ == "__main__":` — never by a spawned DataLoader worker."""
    global DRONE_DIR, NON_DRONE_DIR, CACHE_DIR, GAP, FORCE_REBUILD_CACHE, RESULTS_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--drone_dir", type=str, default="drones",
                         help="Folder containing drone videos")
    parser.add_argument("--non_drone_dir", type=str, default="non-drones",
                         help="Folder containing non-drone videos")
    parser.add_argument("--gap", type=int, default=5,
                         help="Frame gap (in frames) used for the two-frame difference")
    parser.add_argument("--cache_dir", type=str, default="diff_cache",
                         help="Where cached diff frames are stored")
    parser.add_argument("--rebuild_cache", action="store_true",
                         help="Force re-extraction of diff frames even if cache exists")
    args = parser.parse_args()

    DRONE_DIR = args.drone_dir
    NON_DRONE_DIR = args.non_drone_dir
    CACHE_DIR = args.cache_dir
    GAP = args.gap
    FORCE_REBUILD_CACHE = args.rebuild_cache
    RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.pth")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("DRONE vs NON-DRONE  |  TWO-FRAME-DIFFERENCE TRAINING PIPELINE")
    print("=" * 70)
    print(f"Drone videos dir      : {DRONE_DIR}")
    print(f"Non-drone videos dir  : {NON_DRONE_DIR}")
    print(f"Frame gap (diff)      : {GAP}")
    print(f"Diff cache dir        : {CACHE_DIR}")
    print(f"Device                : {DEVICE}")
    print(f"Results will be saved : {RESULTS_PATH}")
    print("=" * 70)


# =====================================================================
# 2. Two-frame-difference cache builder
# =====================================================================
def find_videos(directory):
    if not os.path.isdir(directory):
        return []
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(vids)


def cache_already_built(video_out_dir):
    return os.path.isdir(video_out_dir) and len(os.listdir(video_out_dir)) > 0


def process_video_to_diffs(video_path, label, label_name, cache_root):
    """
    Sequentially decode a video ONCE, maintain a rolling buffer of the last
    (GAP + 1) grayscale frames, and for every eligible frame emit:

        diff = |frame[t] - frame[t - GAP]|

    Applies the asymmetric "too dark / no signal" filtering described in
    the module docstring. Returns the list of saved diff-image paths.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(cache_root, label_name, basename)

    if not FORCE_REBUILD_CACHE and cache_already_built(video_out_dir):
        cached = sorted(glob.glob(os.path.join(video_out_dir, "*.jpg")))
        print(f"    [cache hit] {basename}: {len(cached)} diff frames already cached, skipping decode.")
        return cached, len(cached), 0

    os.makedirs(video_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [WARN] could not open '{video_path}', skipping this video.")
        return [], 0, 0

    buf = deque(maxlen=GAP + 1)
    saved_paths = []
    kept = 0
    skipped_lowsignal = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != IMG_W or gray.shape[0] != IMG_H:
            gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        buf.append(gray)

        if len(buf) == GAP + 1 and (frame_idx % FRAME_STRIDE == 0):
            older, newer = buf[0], buf[-1]
            diff = cv2.absdiff(newer, older)

            if label == 1:
                # DRONE class: require a minimum number of clearly-active
                # pixels, otherwise the drone left no visible motion trace
                # in this diff and the frame would just teach the model
                # "background noise = drone".
                active_px = int(np.count_nonzero(diff > ACTIVE_PIXEL_THRESH))
                keep = active_px >= MIN_ACTIVE_PIXELS_DRONE
            else:
                # NON-DRONE class: keep low-motion frames (they are valid,
                # useful negatives). Only reject essentially-zero /
                # duplicate-frame artifacts.
                keep = int(diff.max()) > MIN_MAX_DIFF_NONDRONE

            if keep:
                out_path = os.path.join(video_out_dir, f"diff_{frame_idx:08d}.jpg")
                cv2.imwrite(out_path, diff)
                saved_paths.append(out_path)
                kept += 1
            else:
                skipped_lowsignal += 1

            if kept >= MAX_FRAMES_PER_VIDEO:
                break

        frame_idx += 1

    cap.release()
    return saved_paths, kept, skipped_lowsignal


def build_dataset_index(drone_dir, non_drone_dir, cache_root):
    """
    Walks both folders, builds the diff-frame cache, and returns a
    per-video registry used later for the stratified train/val split.

    registry[video_basename] = {
        'label': 0 or 1,
        'paths': [list of cached diff-frame file paths],
    }
    """
    print("\n[STAGE 1/4] Building two-frame-difference cache...")
    registry = {}
    totals = {"drone_kept": 0, "drone_skipped": 0, "nondrone_kept": 0, "nondrone_skipped": 0}

    for label, label_name, folder in [(1, "drones", drone_dir), (0, "non-drones", non_drone_dir)]:
        videos = find_videos(folder)
        print(f"\n  -> Scanning '{folder}' ({label_name}): {len(videos)} video(s) found.")
        if len(videos) == 0:
            print(f"     [WARN] No videos found in '{folder}'. Check the path.")
            continue

        for vp in tqdm(videos, desc=f"  Processing {label_name}"):
            paths, kept, skipped = process_video_to_diffs(vp, label, label_name, cache_root)
            if kept == 0:
                print(f"    [WARN] '{os.path.basename(vp)}' produced 0 usable diff frames, excluding from dataset.")
                continue
            basename = os.path.splitext(os.path.basename(vp))[0]
            registry[f"{label_name}/{basename}"] = {"label": label, "paths": paths}

            if label == 1:
                totals["drone_kept"] += kept
                totals["drone_skipped"] += skipped
            else:
                totals["nondrone_kept"] += kept
                totals["nondrone_skipped"] += skipped

    print("\n  --- Cache build summary ---")
    print(f"  Drone frames kept     : {totals['drone_kept']}  "
          f"(skipped as low-signal: {totals['drone_skipped']})")
    print(f"  Non-drone frames kept : {totals['nondrone_kept']}  "
          f"(skipped as degenerate: {totals['nondrone_skipped']})")
    print(f"  Total usable videos   : {len(registry)}")
    return registry


# =====================================================================
# 3. Train/Val split — stratified at the VIDEO level (no leakage)
# =====================================================================
def split_registry(registry, val_fraction, seed):
    print("\n[STAGE 2/4] Splitting videos into train/val (stratified, no frame leakage)...")
    video_keys = list(registry.keys())
    video_labels = [registry[k]["label"] for k in video_keys]

    train_keys, val_keys = train_test_split(
        video_keys,
        test_size=val_fraction,
        random_state=seed,
        stratify=video_labels,
    )

    train_samples = [(p, registry[k]["label"]) for k in train_keys for p in registry[k]["paths"]]
    val_samples = [(p, registry[k]["label"]) for k in val_keys for p in registry[k]["paths"]]

    n_train_pos = sum(1 for _, l in train_samples if l == 1)
    n_train_neg = sum(1 for _, l in train_samples if l == 0)
    n_val_pos = sum(1 for _, l in val_samples if l == 1)
    n_val_neg = sum(1 for _, l in val_samples if l == 0)

    print(f"  Train videos: {len(train_keys)}  |  Val videos: {len(val_keys)}")
    print(f"  Train frames: {len(train_samples)}  (drone={n_train_pos}, non-drone={n_train_neg})")
    print(f"  Val frames  : {len(val_samples)}  (drone={n_val_pos}, non-drone={n_val_neg})")

    if n_train_pos == 0 or n_train_neg == 0:
        raise RuntimeError("Training split is missing one of the two classes entirely — "
                            "check your input folders / filtering thresholds.")

    pos_weight = n_train_neg / n_train_pos
    print(f"  Computed class imbalance -> pos_weight = {pos_weight:.3f}")
    return train_samples, val_samples, pos_weight


# =====================================================================
# 4. Dataset — reads cached diff frames, splits into patches
# =====================================================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])


class DiffPatchDataset(Dataset):
    """
    Loads a pre-computed two-frame-difference image, cuts it into a
    PATCH_GRID x PATCH_GRID grid of patches (each reflection-padded by
    PAD pixels), and returns the stacked patches + a single video-level
    binary label (1 = drone, 0 = non-drone).
    """
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.pad_fn = torch.nn.ReflectionPad2d(PAD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"    [WARN] could not read cached diff frame '{path}', using a blank frame instead.")
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil)  # [1, IMG_H, IMG_W]

        patches = []
        for row in range(PATCH_GRID):
            for col in range(PATCH_GRID):
                x0 = col * PATCH_W
                y0 = row * PATCH_H
                patch = img_tensor[:, y0:y0 + PATCH_H, x0:x0 + PATCH_W]
                patches.append(self.pad_fn(patch))

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return torch.stack(patches), label_tensor  # [16, 1, PATCH_H+2, PATCH_W+2], scalar label


def collate_fn_patch(batch):
    patch_batches, labels = zip(*batch)
    return torch.stack(patch_batches), torch.stack(labels)


# =====================================================================
# 5. Model — PoT (power-of-two) quantization-aware patch classifier
# =====================================================================
class PoTQuantizerSTE(torch.autograd.Function):
    """Straight-through estimator: forward snaps weights to the nearest
    power of two, backward passes gradients through unchanged."""

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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (PoTConv2d, PoTLinear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

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


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_hard_pot_state_dict(current_model):
    """Snap all >=2D weight tensors to hard power-of-two values, leaving
    BatchNorm / bias params untouched. Used to validate on the exact
    weights that will be deployed."""
    pot_state_dict = {}
    with torch.no_grad():
        for name, param in current_model.state_dict().items():
            if "weight" in name and param.dim() >= 2:
                sign = torch.sign(param)
                x_abs = torch.clamp(torch.abs(param), min=2.0 ** -9)
                log2_x = torch.clamp(torch.round(torch.log2(x_abs)), min=-8.0, max=7.0)
                pot_weight = sign * (2.0 ** log2_x)
                pot_weight[torch.abs(param) < (2.0 ** -8)] = 0.0
                pot_state_dict[name] = pot_weight
            else:
                pot_state_dict[name] = param
    return pot_state_dict


# =====================================================================
# 6. MAIN — build data, train, save results.pth
# =====================================================================
def main():
    setup_config()

    registry = build_dataset_index(DRONE_DIR, NON_DRONE_DIR, CACHE_DIR)
    if len(registry) == 0:
        raise RuntimeError("No usable videos found in either folder — nothing to train on.")

    train_samples, val_samples, pos_weight = split_registry(registry, VAL_FRACTION, RANDOM_SEED)

    print("\n[STAGE 3/4] Building datasets and dataloaders...")
    train_ds = DiffPatchDataset(train_samples, transform=train_transform)
    val_ds = DiffPatchDataset(val_samples, transform=val_transform)
    print(f"  Train dataset size: {len(train_ds)}")
    print(f"  Val dataset size  : {len(val_ds)}")

    # NUM_WORKERS: on Windows, each worker is a fresh spawned process that
    # re-imports this file and opens its own shared-memory handles to pass
    # tensors back to the main process. Too many workers exhausts the
    # Windows pagefile-backed shared memory (the "Couldn't open shared file
    # mapping" / error 1455 crash). 2 is a safe default on Windows; Linux
    # can typically handle more (e.g. 4-8) since it uses fork + /dev/shm.
    num_workers = 2 if os.name == "nt" else 4
    persistent_workers = num_workers > 0

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=num_workers, collate_fn=collate_fn_patch,
                           pin_memory=True, persistent_workers=persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_fn_patch,
                         pin_memory=True, persistent_workers=persistent_workers)

    print("\n[STAGE 4/4] Training...")
    model = PatchClassifier6(in_ch=1, base_filters=16).to(DEVICE)
    print(f"  Model initialized on {DEVICE}.")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Translate the class imbalance into a focal-loss alpha (weight on the
    # positive/drone class), clipped to a sane range.
    alpha = max(0.1, min(0.9, pos_weight / (1.0 + pos_weight)))
    print(f"  Focal loss alpha={alpha:.3f} (derived from pos_weight={pos_weight:.3f}), gamma=2.0")

    criterion = BinaryFocalLoss(alpha=alpha, gamma=2.0, reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dl),
        epochs=NUM_EPOCHS, pct_start=0.3,
    )

    best_val_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        print("\n" + "-" * 70)
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        # ---------------- Training ----------------
        model.train()
        total_train_loss = 0.0
        train_preds_list, train_labels_list = [], []

        pbar = tqdm(train_dl, desc=f"  [Train]")
        for patchbatch, labels in pbar:
            patchbatch = patchbatch.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(patchbatch)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_preds_list.append(preds.cpu())
                train_labels_list.append(labels.cpu())

        avg_train_loss = total_train_loss / len(train_dl)
        tp_np = torch.cat(train_preds_list).numpy()
        tl_np = torch.cat(train_labels_list).numpy()
        tn, fp, fn, tp = confusion_matrix(tl_np, tp_np, labels=[0, 1]).ravel()
        print(f"  [Train] avg_loss={avg_train_loss:.4f} | CM(thresh=0.5): TN={tn} FP={fp} FN={fn} TP={tp}")

        # ---------------- Validation (on HARD PoT weights) ----------------
        model.eval()
        fp32_backup = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        hard_pot_dict = get_hard_pot_state_dict(model)
        model.load_state_dict(hard_pot_dict)

        total_val_loss = 0.0
        val_probs_list, val_labels_list = [], []

        pbar_val = tqdm(val_dl, desc=f"  [Val/PoT]")
        with torch.no_grad():
            for patchbatch, labels in pbar_val:
                patchbatch = patchbatch.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                logits = model(patchbatch)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                val_probs_list.append(torch.sigmoid(logits).cpu())
                val_labels_list.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_dl)
        val_probs_np = torch.cat(val_probs_list).numpy()
        val_labels_np = torch.cat(val_labels_list).numpy()
        val_preds_np = (val_probs_np > 0.5).astype(int)

        val_acc = accuracy_score(val_labels_np, val_preds_np)
        val_p = precision_score(val_labels_np, val_preds_np, average="binary", zero_division=0)
        val_r = recall_score(val_labels_np, val_preds_np, average="binary", zero_division=0)
        val_f1 = f1_score(val_labels_np, val_preds_np, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(val_labels_np, val_preds_np, labels=[0, 1]).ravel()

        print(f"  [Val/PoT] avg_loss={avg_val_loss:.4f} | CM: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  [Val/PoT] acc={val_acc:.4f} | precision={val_p:.4f} | recall={val_r:.4f} | F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # model currently holds the hard-PoT weights -> exactly what we want to ship
            torch.save(model.state_dict(), RESULTS_PATH)
            print(f"  ---> New best model! F1={best_val_f1:.4f}. Saved to '{RESULTS_PATH}'")
        else:
            print(f"  (No improvement. Best F1 so far: {best_val_f1:.4f})")

        # restore FP32 weights so training continues correctly next epoch
        model.load_state_dict(fp32_backup)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best validation F1 : {best_val_f1:.4f}")
    print(f"Weights saved to   : {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()