#!/usr/bin/env python3
"""
test_midline_methods.py — Compare midline extraction methods on sample frames.

Both methods use the same background-subtraction binary (blue channel, BG_DIFF_THRESH=40),
so the binarization is identical and the only variable is the midline algorithm:

  A) Column-wise (current pipeline):
       midline_y[x] = (top_y + bottom_y) / 2  for each x-column of the blob

  B) Chain-walking (Python port of MATLAB midlineExtraction.m, Alexandros / Gaetan):
       Walk from head (leftmost blob pixel) in equal arc-length steps,
       following local orientation continuity. Produces N_CLUSTERS+1 joints.

Usage
-----
python test_midline_methods.py --video <path> \\
    [--homography_file <path>] \\
    [--n_frames 6] \\
    [--n_clusters 15] \\
    [--robot_length_px 400]
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from track_robot import (
    apply_homography,
    compute_background,
    detect_robot,
    _detect_swim_direction,
)

# ─────────────────────────────────────────────────────────────────────────────
# Chain-walking midline  (port of midlineExtraction.m)
# ─────────────────────────────────────────────────────────────────────────────

N_CLUSTERS = 15    # MATLAB N_clusters = 15  → 16 joints


def midline_extraction_chain(BW: np.ndarray,
                              robot_length_px: float,
                              n_clusters: int = N_CLUSTERS,
                              head_is_left: bool = True,
                              ) -> np.ndarray | None:
    """
    Walk the robot body from head to tail in equal arc-length steps,
    following local orientation continuity.

    Parameters
    ----------
    BW              : uint8 binary mask, non-zero = robot pixels
    robot_length_px : estimated body length in pixels
    n_clusters      : body segments; returns n_clusters+1 joints
    head_is_left    : if False, start from the rightmost column

    Returns
    -------
    joints : float (n_clusters+1, 2) array of (x, y), or None on failure.
    """
    dr        = robot_length_px / n_clusters
    r_search  = 0.4 * dr
    theta_lim = 0.5 * np.pi          # ±90° angular cone

    ys_all, xs_all = np.where(BW > 0)
    if xs_all.size == 0:
        return None

    # Head: leftmost (or rightmost) blob column, median y in that column
    if head_is_left:
        x_start   = int(xs_all.min())
        init_angle = 0.0             # walk towards +x
    else:
        x_start   = int(xs_all.max())
        init_angle = np.pi           # walk towards -x

    y_at_start = ys_all[xs_all == x_start]
    start = np.array([float(x_start),
                      float(y_at_start[len(y_at_start) // 2])], dtype=float)

    joints    = np.full((n_clusters + 1, 2), np.nan)
    joints[0] = start
    current   = start.copy()
    old_angle = init_angle

    for kj in range(n_clusters):
        dx    = xs_all.astype(float) - current[0]
        dy    = ys_all.astype(float) - current[1]
        r     = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)

        dangle = (theta - old_angle + np.pi) % (2 * np.pi) - np.pi

        sel = np.where(
            (r > dr - r_search) & (r < dr + r_search) & (np.abs(dangle) < theta_lim)
        )[0]

        if sel.size > 1:
            new_angle = np.arctan2(
                ys_all[sel].mean() - current[1],
                xs_all[sel].mean() - current[0],
            )
        elif kj < n_clusters - 2:
            return None          # failed too early — bad frame
        else:
            new_angle = old_angle

        joints[kj + 1] = [current[0] + dr * np.cos(new_angle),
                           current[1] + dr * np.sin(new_angle)]
        current   = joints[kj + 1]
        old_angle = new_angle

    return joints


def estimate_robot_length(BW: np.ndarray) -> float | None:
    """95 % of the x-span of the blob as a proxy for body length."""
    ys, xs = np.where(BW > 0)
    if xs.size == 0:
        return None
    return 0.95 * float(xs.max() - xs.min())


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(video_path: str,
                   H:               np.ndarray | None = None,
                   n_frames:        int   = 6,
                   n_clusters:      int   = N_CLUSTERS,
                   robot_length_px: float | None = None):

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("Computing background model...")
    background = compute_background(video_path, H=H)

    head_left = _detect_swim_direction(video_path, background, H=H)
    print(f"Swimming direction: {'left→right' if head_left else 'right→left'}")

    frame_indices = np.linspace(int(0.1 * total), int(0.9 * total),
                                 n_frames, dtype=int)

    fig, axes = plt.subplots(n_frames, 3, figsize=(20, 4 * n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    for row, fi in enumerate(frame_indices):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        if H is not None:
            frame = apply_homography(frame, H)

        t = fi / fps

        # ── Method A: BG-sub binary + column-wise midline ─────────────────
        det_A = detect_robot(frame, background=background, debug=True,
                             midline_method="colwise")

        # ── Method B: same blob mask → chain-walking ──────────────────────
        det_B = detect_robot(frame, background=background, debug=True,
                             midline_method="chain",
                             robot_length_px=robot_length_px,
                             head_is_left=head_left)
        joints = det_B.joints

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Panel 1: BG-sub mask + col-wise midline ───────────────────────
        ax = axes[row, 0]
        ax.imshow(frame_rgb)
        if det_A.mask is not None:
            ax.imshow(det_A.mask, alpha=0.45, cmap="Greens", vmin=0, vmax=255)
        if det_A.detected:
            ax.plot(det_A.midline_xs, det_A.midline_ys, 'c-', lw=1.5)
            ax.plot(det_A.cx, det_A.cy, 'g*', ms=12)
        ax.set_title(f"t={t:.1f}s  A: col-wise midline", fontsize=9)
        ax.axis("off")

        # ── Panel 2: same mask + chain midline ────────────────────────────
        ax = axes[row, 1]
        ax.imshow(frame_rgb)
        if det_B.mask is not None:
            ax.imshow(det_B.mask, alpha=0.35, cmap="Oranges", vmin=0, vmax=255)
        if joints is not None:
            valid = ~np.isnan(joints[:, 0])
            ax.plot(joints[valid, 0], joints[valid, 1], 'r.-', lw=2, ms=6)
            ax.plot(joints[valid, 0][0],  joints[valid, 1][0],
                    'rs', ms=10, label="head")
            ax.plot(joints[valid, 0][-1], joints[valid, 1][-1],
                    'r^', ms=10, label="tail")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"t={t:.1f}s  B: chain-walking midline", fontsize=9)
        ax.axis("off")

        # ── Panel 3: overlay ──────────────────────────────────────────────
        ax = axes[row, 2]
        ax.imshow(frame_rgb)
        if det_A.detected:
            ax.plot(det_A.midline_xs, det_A.midline_ys,
                    'c-', lw=2.5, label="A col-wise", alpha=0.9)
            ax.plot(det_A.cx, det_A.cy, 'c*', ms=12)
        if joints is not None:
            valid = ~np.isnan(joints[:, 0])
            ax.plot(joints[valid, 0], joints[valid, 1],
                    'r-', lw=2, label="B chain", alpha=0.9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"t={t:.1f}s  Overlay", fontsize=9)
        ax.axis("off")

        print(f"  frame {fi:4d}  t={t:.1f}s  "
              f"A:{'ok' if det_A.detected else 'MISS'}  "
              f"B:{'ok' if joints is not None else 'MISS'}")

    fig.suptitle(
        f"{os.path.basename(video_path)}\n"
        "Both use the same BG-subtraction mask  |  "
        "A (cyan): column-wise midline  |  B (red): chain-walking midline",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = os.path.splitext(video_path)[0] + "_midline_comparison.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive robot-length picker
# ─────────────────────────────────────────────────────────────────────────────

def pick_robot_length(video_path: str, H: np.ndarray | None = None) -> float:
    """
    Show a frame from the middle of the video and ask the user to click
    the head and tail of the robot.  Returns the pixel distance between them.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total//2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame from video")

    if H is not None:
        frame = apply_homography(frame, H)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(frame_rgb)
    ax.set_title("Click HEAD then TAIL of the robot  (close window when done)",
                 fontsize=12)
    ax.axis("off")

    clicks = []
    markers = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        clicks.append((x, y))
        label = ["HEAD", "TAIL"][min(len(clicks) - 1, 1)]
        m, = ax.plot(x, y, 'r+', ms=18, mew=3)
        ax.annotate(label, (x, y), color='red', fontsize=11,
                    xytext=(8, 8), textcoords='offset points')
        markers.append(m)
        if len(clicks) == 2:
            ax.plot([clicks[0][0], clicks[1][0]],
                    [clicks[0][1], clicks[1][1]],
                    'r--', lw=1.5)
            dist = np.hypot(clicks[1][0] - clicks[0][0],
                            clicks[1][1] - clicks[0][1])
            ax.set_title(f"Robot length = {dist:.1f} px  (close window to confirm)",
                         fontsize=12)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

    if len(clicks) < 2:
        raise RuntimeError("Need exactly 2 clicks (head and tail)")

    dist = float(np.hypot(clicks[1][0] - clicks[0][0],
                          clicks[1][1] - clicks[0][1]))
    print(f"Robot length picked: {dist:.1f} px")
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video",            required=True)
    p.add_argument("--homography_file",  default=None)
    p.add_argument("--n_frames",         type=int,   default=6)
    p.add_argument("--n_clusters",       type=int,   default=N_CLUSTERS,
                   help=f"Body segments for chain midline (default {N_CLUSTERS})")
    p.add_argument("--robot_length_px",  type=float, default=None,
                   help="Robot body length in pixels; auto-estimated if not given")
    p.add_argument("--pick_length",      action="store_true",
                   help="Interactively pick robot head+tail to measure length in px")
    args = p.parse_args()

    H = None
    if args.homography_file:
        H = np.load(args.homography_file)
        print(f"Loaded homography from {args.homography_file}")

    robot_length_px = args.robot_length_px
    if args.pick_length:
        robot_length_px = pick_robot_length(args.video, H=H)

    run_comparison(
        args.video,
        H               = H,
        n_frames        = args.n_frames,
        n_clusters      = args.n_clusters,
        robot_length_px = robot_length_px,
    )


if __name__ == "__main__":
    main()
