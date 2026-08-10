"""
test_frame_diff.py
-------------------
Quick, throwaway script to VISUALIZE frame differencing on a video before
we commit to changing the training pipeline.

It does NOT train anything. It just saves side-by-side images so you can
eyeball whether a drone/bird survives as a visible blob after differencing.

Usage:
    python test_frame_diff.py --video abc.mp4 --gap 3 --start 100 --num 10
"""

import cv2
import numpy as np
import os
import argparse


def read_gray_frame(cap, frame_idx):
    """Jump to a specific frame number and read it as grayscale."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def two_frame_diff(frame_a, frame_b):
    """
    Simplest possible motion signal: |A - B|.
    Anything that didn't move (sky, trees, buildings) cancels out to
    near-zero. Anything that moved (drone, bird) shows up bright.

    Downside: a moving point object leaves TWO faint blobs (one from
    each frame's edge), a bit like a ghost/echo instead of one clean dot.
    """
    diff = cv2.absdiff(frame_a, frame_b)
    return diff


def three_frame_diff(frame_prev, frame_curr, frame_next):
    """
    Cleaner motion signal that fixes the 'ghosting' problem above.

    Idea: compute two diffs -
        diff1 = |curr - prev|   (motion between previous and current)
        diff2 = |curr - next|   (motion between current and next)
    Take the pixel-wise MINIMUM of the two.

    Why minimum works: a real moving object shows up bright in BOTH
    diffs (because it's different from both its neighbors in time).
    Ghost/edge artifacts from a single diff usually only show up
    strongly in ONE of the two. Taking the minimum keeps only pixels
    that agree across both comparisons -> one clean blob at the
    object's CURRENT position, not a smear.
    """
    diff1 = cv2.absdiff(frame_curr, frame_prev)
    diff2 = cv2.absdiff(frame_curr, frame_next)
    combined = cv2.min(diff1, diff2)
    return combined


def denoise(diff_img, blur_ksize=3, threshold=None):
    """
    Optional cleanup:
    - Slight blur smooths out JPEG compression blockiness that can
      look like fake 'motion' in flat areas (sky).
    - Threshold zeroes out very faint differences (sensor/compression
      noise floor) so only real motion remains visible.
    """
    out = cv2.GaussianBlur(diff_img, (blur_ksize, blur_ksize), 0)
    if threshold is not None:
        _, out = cv2.threshold(out, threshold, 255, cv2.THRESH_TOZERO)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="abc.mp4", help="Path to input video")
    parser.add_argument("--gap", type=int, default=3, help="Frame gap between A and B (in FRAMES, not ms)")
    parser.add_argument("--start", type=int, default=0, help="Frame index to start sampling from")
    parser.add_argument("--num", type=int, default=10, help="Number of sample points to visualize")
    parser.add_argument("--out_dir", type=str, default="diff_preview", help="Where to save preview images")
    parser.add_argument("--threshold", type=int, default=15, help="Noise floor threshold (0-255), None-like via -1 to disable")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found at '{args.video}'. Put abc.mp4 next to this script or pass --video <path>.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video '{args.video}'.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {args.video} | {total_frames} frames | {fps:.1f} fps")
    print(f"Frame gap = {args.gap} frames (~{1000*args.gap/max(fps,1):.1f} ms at this fps)")

    threshold = None if args.threshold < 0 else args.threshold
    saved = 0

    for i in range(args.num):
        center = args.start + i * (args.gap * 3)  # space out samples across the video
        prev_idx = center - args.gap
        curr_idx = center
        next_idx = center + args.gap

        if next_idx >= total_frames or prev_idx < 0:
            continue

        f_prev = read_gray_frame(cap, prev_idx)
        f_curr = read_gray_frame(cap, curr_idx)
        f_next = read_gray_frame(cap, next_idx)

        if f_prev is None or f_curr is None or f_next is None:
            continue

        diff2 = two_frame_diff(f_prev, f_curr)
        diff3 = three_frame_diff(f_prev, f_curr, f_next)

        diff2_clean = denoise(diff2, threshold=threshold)
        diff3_clean = denoise(diff3, threshold=threshold)

        # Stack: raw current frame | 2-frame diff | 3-frame diff (all cleaned)
        # Boost brightness of diffs so faint blobs are actually visible to your eye
        diff2_vis = cv2.convertScaleAbs(diff2_clean, alpha=3.0)
        diff3_vis = cv2.convertScaleAbs(diff3_clean, alpha=3.0)

        row = np.hstack([f_curr, diff2_vis, diff3_vis])
        label = f"frame_{curr_idx:06d}  |  raw  -  2frame_diff  -  3frame_diff"
        row_bgr = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
        cv2.putText(row_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = os.path.join(args.out_dir, f"preview_{curr_idx:06d}.jpg")
        cv2.imwrite(out_path, row_bgr)
        saved += 1

    cap.release()
    print(f"\nSaved {saved} preview images to '{args.out_dir}/'.")
    print("Open a few and check: does the drone/bird show up as a clear blob")
    print("in the diff panels, especially when it's moving slowly or hovering?")


if __name__ == "__main__":
    main()
