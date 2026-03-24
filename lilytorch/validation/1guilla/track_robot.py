"""
Robot speed tracking via HSV color segmentation (top-down pool camera).

Workflow
--------
1. (Optional) Perspective-correct the video first:

    python track_robot.py --homography --video <path>
      → picks 4 corners interactively, saves <video>_homography.npy,
        then tracks the corrected frames.

2. Track a single video (homography already computed):

    python track_robot.py --video <path> --meters_per_pixel 0.00123
      --homography_file <path>_homography.npy

3. Batch-process all videos from swim_experiment_summary.csv:

    python track_robot.py --meters_per_pixel 0.00123
      [--homography_file shared_homography.npy]

4. Calibrate scale interactively (click two points on a frame):

    python track_robot.py --calibrate --video <path>
      [--homography_file <path>_homography.npy]

5. Tune HSV thresholds interactively (if lighting differs):

    python track_robot.py --tune --video <path>

Data files are expected at:
    <script_dir>/../../farms_examples/../  (adjust --summary_csv / --video_dir)

Defaults assume the script lives next to a `videos/` folder and
`swim_experiment_summary.csv` (see --summary_csv / --video_dir).
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt

# Add process_videos to path so we can import video_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "process_videos"))
from video_utils import pick_four_points, pick_sync_frame, compute_homography  # noqa: E402

# ── Blue-channel detection parameters ────────────────────────────────────────
# The pool is light blue (B ≈ 140); the robot body (orange-yellow) and its
# black connectors both have low blue values (B < 100), so a simple blue-channel
# threshold captures the full robot as a single solid blob — including parts
# that the HSV approach missed.
BLUE_THRESH   = 110    # raw B-channel threshold: robot (B≈0–90) vs pool water (B≈128+)
BOX_WINDOW    = 25     # homogeneous (box) filter window for edge smoothing
SMOOTH_THRESH = 0.45   # after blurring, keep pixels with value ≥ this

# Background subtraction threshold: how much darker a pixel must be vs the
# median background to count as "robot".  Fixed fiducials / pool floor cancel
# out; only the moving robot body produces large differences (~50–90 px).
BG_DIFF_THRESH = 40    # pixels with |frame_B - bg_B| > this are robot candidates

# Inner water ROI.
# Top rail (white, B<125) occupies y=85–124 → start at 125 to exclude it.
ROI_X1, ROI_X2 = 130, 1790
ROI_Y1, ROI_Y2 = 130, 710

# Minimum blob area (px²) to be a valid detection
MIN_AREA_PX = 800

# Maximum plausible centroid displacement between consecutive frames (pixels).
MAX_JUMP_PX = 60

# Chain-walking midline (port of MATLAB midlineExtraction.m)
N_CLUSTERS = 15    # body segments → N_CLUSTERS+1 joints

# ─────────────────────────────────────────────────────────────────────────────
# Homography helpers
# ─────────────────────────────────────────────────────────────────────────────

def interactive_homography(video_path: str) -> np.ndarray:
    """
    Let the user pick a reference frame and 4 rectangle corners, compute and
    save the perspective-correction homography alongside the video.

    Returns the H matrix (3×3).
    """
    print("\n[Homography] Pick the reference frame (a clear view of the pool).")
    ref_idx = pick_sync_frame(video_path, "Homography — pick reference frame")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Cannot read frame {ref_idx}")

    print("\n[Homography] Click the 4 corners of a known rectangle")
    print("  Order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")
    corners = pick_four_points(frame, "Homography — click 4 rectangle corners")

    H, out_w, out_h = compute_homography(frame, corners)
    print(f"  Warp: {frame.shape[1]}×{frame.shape[0]} → {out_w}×{out_h}")

    save_path = os.path.splitext(video_path)[0] + "_homography.npy"
    np.save(save_path, H)
    print(f"  Saved → {save_path}")
    return H


def load_homography(path: str) -> np.ndarray:
    H = np.load(path)
    print(f"[Homography] Loaded from {path}")
    return H


def apply_homography(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp a single frame with a precomputed homography."""
    # Compute output size from the image corners
    h, w = frame.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    out_w = int(np.ceil(warped_corners[:, 0].max() - warped_corners[:, 0].min()))
    out_h = int(np.ceil(warped_corners[:, 1].max() - warped_corners[:, 1].min()))
    return cv2.warpPerspective(frame, H, (out_w, out_h))


# ─────────────────────────────────────────────────────────────────────────────
# Background model
# ─────────────────────────────────────────────────────────────────────────────

def compute_background(video_path: str,
                       H: np.ndarray | None = None,
                       n_samples: int = 40) -> np.ndarray:
    """
    Pixel-wise median of the B channel across N evenly-spaced frames.
    The robot averages out (different position each frame); fixed objects
    (fiducials, pool floor metal) remain in the background and are subtracted.
    Returns a uint8 image the same size as the video frames.
    """
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stack = []
    for idx in np.linspace(0, total - 1, n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if H is not None:
            frame = apply_homography(frame, H)
        stack.append(frame[:, :, 0].astype(np.float32))
    cap.release()
    if not stack:
        sys.exit(f"[ERROR] Could not read any frames from: {video_path}")
    bg = np.median(stack, axis=0).astype(np.uint8)
    print(f"  Background computed from {len(stack)} frames.")
    return bg


# ─────────────────────────────────────────────────────────────────────────────
# Core detection
# ─────────────────────────────────────────────────────────────────────────────



def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class Detection:
    """Result of detect_robot."""
    __slots__ = ("cx", "cy", "area", "midline_xs", "midline_ys",
                 "midline_p1", "midline_p2", "mask", "contour", "joints",
                 "head_pt")

    def __init__(self, cx, cy, area,
                 midline_xs=None, midline_ys=None,
                 midline_p1=None, midline_p2=None,
                 mask=None, contour=None, joints=None, head_pt=None):
        self.cx, self.cy, self.area = cx, cy, area
        self.midline_xs = midline_xs  # 1-D array: x coords of midline points
        self.midline_ys = midline_ys  # 1-D array: y coords of midline points
        self.midline_p1 = midline_p1  # (x, y) head end of midline
        self.midline_p2 = midline_p2  # (x, y) tail end of midline
        self.mask    = mask           # binary mask (only when debug=True)
        self.contour = contour        # largest contour (only when debug=True)
        self.joints  = joints         # (N_CLUSTERS+1, 2) chain joints or None
        self.head_pt = head_pt        # (x, y) blob head — always set when blob valid

    @property
    def detected(self):
        return self.cx is not None


_roi_cache: dict = {}


def _get_roi(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _roi_cache:
        m = np.zeros((h, w), np.uint8)
        m[ROI_Y1:min(ROI_Y2, h), ROI_X1:min(ROI_X2, w)] = 255
        _roi_cache[key] = m
    return _roi_cache[key]


def _extract_midline(blob_mask: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extract the robot midline from a filled binary blob.

    For each x-column that contains robot pixels, the midline y is the average
    of the topmost and bottommost robot pixel in that column.  This directly
    implements the paper's method: "averaging the values of the robot's body
    lateral sides that correspond to the same longitudinal coordinate".

    Returns (xs, ys) as 1-D arrays sorted by x, or (None, None) if too few
    columns are occupied.
    """
    ys_all, xs_all = np.where(blob_mask)
    if xs_all.size == 0:
        return None, None
    x_min, x_max = xs_all.min(), xs_all.max()
    ml_xs, ml_ys = [], []
    for x in range(x_min, x_max + 1):
        col_ys = ys_all[xs_all == x]
        if col_ys.size == 0:
            continue
        ml_xs.append(x)
        ml_ys.append((col_ys.min() + col_ys.max()) / 2.0)
    if len(ml_xs) < 3:
        return None, None
    return np.array(ml_xs, dtype=float), np.array(ml_ys, dtype=float)


def _estimate_robot_length(video_path: str,
                           background: np.ndarray,
                           H: np.ndarray | None = None,
                           n_samples: int = 20) -> float | None:
    """
    Estimate robot body length in pixels by taking the median colwise-midline
    arc-length across N evenly-spaced frames.

    Uses arc-length (sum of segment lengths along the midline curve) rather
    than x-span so the estimate is correct even when the body is curved.
    Returns None if fewer than 3 valid samples are found.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lengths = []
    for idx in np.linspace(int(0.1 * total), int(0.9 * total),
                           n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if H is not None:
            frame = apply_homography(frame, H)
        det = detect_robot(frame, background=background, midline_method="colwise")
        if det.midline_xs is not None and len(det.midline_xs) > 2:
            arc = float(np.hypot(np.diff(det.midline_xs),
                                  np.diff(det.midline_ys)).sum())
            if arc > 0:
                lengths.append(arc)
    cap.release()
    if len(lengths) < 3:
        return None
    est = float(np.median(lengths))
    print(f"  Robot length estimated: {est:.1f} px  (median of {len(lengths)} frames)")
    return est


def _midline_chain(blob_mask: np.ndarray,
                   robot_length_px: float | None = None,
                   n_clusters: int = N_CLUSTERS,
                   head_is_left: bool = True,
                   ) -> np.ndarray | None:
    """
    Chain-walking midline extraction (port of MATLAB midlineExtraction.m).

    Walks from head to tail in equal arc-length steps following local
    orientation continuity.  Returns (n_clusters+1, 2) float array of
    (x, y) joint positions, or None if the chain fails.
    """
    ys_all, xs_all = np.where(blob_mask > 0)
    if xs_all.size == 0:
        return None

    # Estimate robot length from blob x-span if not provided
    rlen = robot_length_px if robot_length_px else 0.95 * float(xs_all.max() - xs_all.min())
    if rlen < 1:
        return None

    dr        = rlen / n_clusters
    r_search  = 0.4 * dr
    theta_lim = 0.5 * np.pi          # ±90° angular cone

    if head_is_left:
        x_start    = int(xs_all.min())
        init_angle = 0.0
    else:
        x_start    = int(xs_all.max())
        init_angle = np.pi

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
            new_angle = np.arctan2(ys_all[sel].mean() - current[1],
                                   xs_all[sel].mean() - current[0])
        elif kj < n_clusters - 2:
            return None
        else:
            new_angle = old_angle
        joints[kj + 1] = [current[0] + dr * np.cos(new_angle),
                           current[1] + dr * np.sin(new_angle)]
        current   = joints[kj + 1]
        old_angle = new_angle

    return joints


def _detect_swim_direction(video_path: str,
                           background: np.ndarray,
                           H: np.ndarray | None = None) -> bool:
    """
    Return True if robot swims left→right (head at left).
    Compares blob centroid at 40 % and 60 % of the video.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cxs   = []
    for frac in (0.4, 0.6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * total))
        ret, frame = cap.read()
        if not ret:
            continue
        if H is not None:
            frame = apply_homography(frame, H)
        det = detect_robot(frame, background=background, midline_method="colwise")
        if det.detected:
            cxs.append(det.cx)
    cap.release()
    if len(cxs) == 2:
        return cxs[1] < cxs[0]   # cx decreasing → moving left → head is at left (leading edge)
    return True


def detect_robot(frame: np.ndarray,
                 background: np.ndarray | None = None,
                 debug: bool = False,
                 midline_method: str = "chain",
                 robot_length_px: float | None = None,
                 head_is_left: bool = True) -> Detection:
    """
    Detect the robot using a blue-channel threshold (paper method).

    Pipeline
    --------
    1. Build binary image from the B channel:
       - With background: pixels where bg_B - frame_B > BG_DIFF_THRESH.
       - Without background: simple threshold B < BLUE_THRESH.
    2. Box filter BOX_WINDOW × BOX_WINDOW, threshold at SMOOTH_THRESH.
    3. Restrict to inner water ROI, keep the largest connected blob.
    4. Extract midline via chain-walking (default) or column-wise.
       chain  : walk from head to tail in equal arc-length steps (N_CLUSTERS+1 joints)
       colwise: for each x-column average top and bottom edges
    5. Centroid = mean of midline points.

    Parameters
    ----------
    midline_method   : "chain" (default) or "colwise"
    robot_length_px  : robot body length in pixels for chain method
                       (auto-estimated from blob x-span if None)
    head_is_left     : True if robot head is at the left of the frame
                       (used by chain method to set walk direction)
    """
    h, w = frame.shape[:2]
    roi  = _get_roi(h, w)

    # ── step 1: binary image ──────────────────────────────────────────────
    blue = frame[:, :, 0].astype(np.float32)
    if background is not None:
        bg     = background.astype(np.float32)
        binary = (bg - blue > BG_DIFF_THRESH).astype(np.float32)
    else:
        binary = (blue < BLUE_THRESH).astype(np.float32)

    # ── step 2: box-filter + intensity threshold ──────────────────────────
    blurred = cv2.boxFilter(binary, ddepth=-1,
                            ksize=(BOX_WINDOW, BOX_WINDOW), normalize=True)
    final = ((blurred >= SMOOTH_THRESH) * 255).astype(np.uint8)

    # ── step 3: ROI + largest blob ────────────────────────────────────────
    final = cv2.bitwise_and(final, roi)
    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return Detection(None, None, 0.0,
                         mask=np.zeros((h, w), np.uint8) if debug else None)

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_AREA_PX:
        return Detection(None, None, area,
                         mask=np.zeros((h, w), np.uint8) if debug else None)

    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [best], -1, 255, -1)

    # ── head point: extreme blob pixel, independent of midline ───────────
    _ys_b, _xs_b = np.where(blob > 0)
    _x_head = int(_xs_b.min() if head_is_left else _xs_b.max())
    _y_cols  = _ys_b[_xs_b == _x_head]
    head_pt  = (int(_x_head), int(_y_cols[len(_y_cols) // 2]))

    # ── step 4: midline extraction ────────────────────────────────────────
    joints = None
    if midline_method == "chain":
        joints = _midline_chain(blob, robot_length_px=robot_length_px,
                                head_is_left=head_is_left)
        if joints is not None:
            valid  = ~np.isnan(joints[:, 0])
            ml_xs  = joints[valid, 0]
            ml_ys  = joints[valid, 1]
        else:
            ml_xs, ml_ys = None, None
    else:
        ml_xs, ml_ys = _extract_midline(blob)

    if ml_xs is None:
        return Detection(None, None, area,
                         mask=blob if debug else None,
                         contour=best if debug else None,
                         head_pt=head_pt)

    # ── step 5: reference position ────────────────────────────────────────
    # Use the blob head_pt (extreme pixel) for x — always defined, never
    # depends on chain success.  For y, use body median if chain succeeded.
    cx = float(head_pt[0])
    if midline_method == "chain" and joints is not None:
        valid_j = joints[~np.isnan(joints[:, 0])]
        cy = float(np.median(valid_j[:, 1]))
    else:
        cy = float(ml_ys.mean())

    p1 = (int(ml_xs[0]),  int(ml_ys[0]))   # head end of midline
    p2 = (int(ml_xs[-1]), int(ml_ys[-1]))  # tail end of midline

    return Detection(cx, cy, area,
                     midline_xs=ml_xs, midline_ys=ml_ys,
                     midline_p1=p1, midline_p2=p2,
                     mask=blob if debug else None,
                     contour=best if debug else None,
                     joints=joints,
                     head_pt=head_pt)


# ─────────────────────────────────────────────────────────────────────────────
# Debug preview
# ─────────────────────────────────────────────────────────────────────────────

def _draw_midline(vis: np.ndarray, det: "Detection",
                  head_trail: list | None = None) -> None:
    """Draw head trail and prominent head marker."""
    # head trail
    if head_trail and len(head_trail) > 1:
        for i in range(len(head_trail) - 1):
            alpha = (i + 1) / len(head_trail)
            color = (0, int(255 * alpha), int(180 * alpha))
            cv2.line(vis, head_trail[i], head_trail[i + 1], color, 2)

    # prominent head marker — uses head_pt (blob extreme, always reliable)
    if det.head_pt is not None:
        hx, hy = det.head_pt
        cv2.circle(vis, (hx, hy), 10, (0, 255, 80), -1)
        cv2.circle(vis, (hx, hy), 10, (0, 0, 0), 2)
        cv2.putText(vis, "H", (hx + 12, hy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)


def preview_video(video_path: str,
                  H: np.ndarray | None = None,
                  save_path: str | None = None,
                  robot_length_px: float | None = None,
                  midline_method: str = "chain"):
    """
    Show (and optionally save) a side-by-side debug view:
      Left  — original frame with midline + centroid overlaid
      Right — binary mask (green = robot pixels)

    Controls
    --------
    SPACE      pause / resume
    n / →      step one frame forward (while paused)
    p / ←      step one frame backward (while paused)
    q / ESC    quit
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    headless = save_path is not None and not _has_display()

    print("  Computing background model...")
    background = compute_background(video_path, H=H)
    print("  Detecting swim direction...")
    head_left = _detect_swim_direction(video_path, background, H=H)
    print(f"  Direction: {'left→right' if head_left else 'right→left'}")
    if robot_length_px is None and midline_method == "chain":
        print("  Estimating robot length...")
        robot_length_px = _estimate_robot_length(video_path, background, H=H)

    # ── pre-compute full head trajectory ──────────────────────────────────
    # Run a fast detection pass (no debug mask) through all frames so the
    # head trail is available from frame 0 — including when scrubbing.
    print("  Pre-computing head trajectory...")
    all_heads: list[tuple[int, int] | None] = []
    _cap = cv2.VideoCapture(video_path)
    _idx = 0
    while True:
        ret, _frame = _cap.read()
        if not ret:
            break
        if H is not None:
            _frame = apply_homography(_frame, H)
        _det = detect_robot(_frame, background=background,
                            midline_method=midline_method,
                            robot_length_px=robot_length_px,
                            head_is_left=head_left)
        all_heads.append(_det.head_pt)
        _idx += 1
        if _idx % 100 == 0:
            print(f"  {_idx}/{total}", end="\r", flush=True)
    _cap.release()
    print(f"  Head trajectory computed ({sum(p is not None for p in all_heads)}"
          f"/{total} frames detected)")

    # ── quadratic fit trajectory (same method as track_video fwd velocity) ─
    _ts_all = np.arange(total, dtype=float) / fps
    _hx_raw = np.array([p[0] if p is not None else np.nan for p in all_heads], dtype=float)
    _hy_raw = np.array([p[1] if p is not None else np.nan for p in all_heads], dtype=float)
    _valid  = ~(np.isnan(_hx_raw) | np.isnan(_hy_raw))
    if _valid.sum() >= 3:
        _px = np.polyfit(_ts_all[_valid], _hx_raw[_valid], 2)
        _py = np.polyfit(_ts_all[_valid], _hy_raw[_valid], 2)
        _hx_fit = np.polyval(_px, _ts_all)
        _hy_fit = np.polyval(_py, _ts_all)
    else:
        _hx_fit, _hy_fit = _hx_raw, _hy_raw
    smooth_path: list[tuple[int, int]] = [
        (int(round(_hx_fit[i])), int(round(_hy_fit[i])))
        for i in range(total)
        if not (np.isnan(_hx_fit[i]) or np.isnan(_hy_fit[i]))
    ]

    TRAIL_LEN  = int(fps * 1.5)   # ~1.5 s
    head_trail: list[tuple[int, int]] = []

    writer    = None
    paused    = False
    frame_idx = 0

    if not headless:
        cv2.namedWindow("Detection preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection preview", 1280, 360)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if H is not None:
            frame = apply_homography(frame, H)

        det = detect_robot(frame, background=background, debug=True,
                           midline_method=midline_method,
                           robot_length_px=robot_length_px,
                           head_is_left=head_left)

        # ── annotated frame ───────────────────────────────────────────────
        vis = frame.copy()

        # smoothed full trajectory (white line) — same path used for fwd velocity
        if len(smooth_path) > 1:
            pts = np.array(smooth_path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 255), thickness=1,
                          lineType=cv2.LINE_AA)

        trail_start = max(0, frame_idx - TRAIL_LEN)
        head_trail = [p for p in all_heads[trail_start:frame_idx + 1]
                      if p is not None]
        _draw_midline(vis, det, head_trail=head_trail)

        # ROI boundary box
        h_vis = vis.shape[0]
        cv2.rectangle(vis, (ROI_X1, ROI_Y1),
                      (min(ROI_X2, vis.shape[1]-1), min(ROI_Y2, h_vis-1)),
                      (0, 200, 255), 1)

        status = f"#{frame_idx}/{total-1}  area={det.area:.0f}px"
        if not det.detected:
            status += "  [NOT DETECTED]"
            color = (0, 0, 255)
        else:
            status += f"  cx={det.cx:.0f} cy={det.cy:.0f}"
            color = (0, 255, 0)
        cv2.putText(vis, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if paused:
            cv2.putText(vis, "PAUSED  (SPACE=play  n/p=step  q=quit)",
                        (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── mask panel ────────────────────────────────────────────────────
        mask = det.mask if det.mask is not None else np.zeros(frame.shape[:2], np.uint8)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[mask > 0] = (0, 220, 0)

        # ── resize both to same height and stack horizontally ─────────────
        target_h = 360
        def _resize_h(img, h):
            scale = h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), h))

        left  = _resize_h(vis, target_h)
        right = _resize_h(mask_bgr, target_h)
        combined = np.hstack([left, right])

        cv2.imshow("Detection preview", combined)

        # ── optional video save ───────────────────────────────────────────
        if save_path is not None:
            if writer is None:
                h_out, w_out = combined.shape[:2]
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w_out, h_out))
            writer.write(combined)

        if headless:
            frame_idx += 1
            if frame_idx >= total:
                break
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total}", end="\r", flush=True)
            continue

        # ── key handling ──────────────────────────────────────────────────
        delay = 0 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('n'), 83) and paused:          # n or →
            frame_idx = min(frame_idx + 1, total - 1)
            continue
        elif key in (ord('p'), 81) and paused:          # p or ←
            frame_idx = max(frame_idx - 1, 0)
            continue

        if not paused:
            frame_idx = min(frame_idx + 1, total - 1)
            if frame_idx == total - 1:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Preview video saved → {save_path}")
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Track a single video
# ─────────────────────────────────────────────────────────────────────────────

def _reject_outliers(xs: np.ndarray, ys: np.ndarray,
                     max_jump: float = MAX_JUMP_PX) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace centroid positions where the frame-to-frame displacement exceeds
    max_jump pixels with NaN, then linearly interpolate over the gaps.
    """
    xs, ys = xs.copy(), ys.copy()
    prev_x, prev_y = None, None
    for i in range(len(xs)):
        if np.isnan(xs[i]):
            prev_x, prev_y = None, None
            continue
        if prev_x is not None:
            jump = np.hypot(xs[i] - prev_x, ys[i] - prev_y)
            if jump > max_jump:
                xs[i] = np.nan
                ys[i] = np.nan
                continue
        prev_x, prev_y = xs[i], ys[i]
    # Linear interpolation over NaN gaps
    idx = np.arange(len(xs))
    valid = ~np.isnan(xs)
    if valid.sum() > 1:
        xs = np.interp(idx, idx[valid], xs[valid])
        ys = np.interp(idx, idx[valid], ys[valid])
    return xs, ys


def _smooth(arr: np.ndarray, fps: float, window_s: float = 0.5) -> np.ndarray:
    """
    Savitzky-Golay filter. Window = window_s seconds (rounded to odd integer).
    Falls back to the raw array if too few points.
    """
    wlen = max(5, int(round(window_s * fps)) | 1)   # ensure odd
    wlen = min(wlen, len(arr) - (1 if len(arr) % 2 == 0 else 0))
    if wlen < 5 or len(arr) < wlen:
        return arr
    return savgol_filter(arr, window_length=wlen, polyorder=3)


def _lowpass(arr: np.ndarray, fps: float, cutoff_hz: float = 3.0) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth low-pass filter.
    NaN gaps are linearly interpolated before filtering, then restored.
    Falls back to the raw array if too few valid points.
    """
    valid = ~np.isnan(arr)
    if valid.sum() < 10:
        return arr
    nyq = 0.5 * fps
    wn  = cutoff_hz / nyq
    if wn >= 1.0:
        return arr
    b, a = butter(4, wn, btype="low")
    # interpolate over NaN so filtfilt doesn't see gaps
    idx = np.arange(len(arr))
    arr_filled = np.interp(idx, idx[valid], arr[valid])
    filtered = filtfilt(b, a, arr_filled)
    filtered[~valid] = np.nan
    return filtered


def track_video(video_path: str, meters_per_pixel: float,
                H: np.ndarray | None = None,
                save_annotated: bool = False,
                robot_length_px: float | None = None,
                midline_method: str = "chain") -> pd.DataFrame:
    """
    Track the robot centroid through every frame.

    Pipeline
    --------
    1. Detect swim direction once (compare centroid at 40 % and 60 % of video).
    2. Detect robot each frame using chain-walking midline.
    3. Reject outlier jumps > MAX_JUMP_PX and interpolate.
    4. Savitzky-Golay smooth positions (0.5 s window).
    5. Differentiate smoothed positions for speed.
    6. Compute two speed columns:
       - speed_2d_mps : full √(dx²+dy²)/dt  (includes lateral undulation)
       - speed_fwd_mps: speed projected onto the mean swimming direction
                        (removes anguilliform y-undulation noise — use this one)

    Returns a DataFrame with columns:
        frame, time_s, x_px, y_px, x_m, y_m, x_sm, y_sm,
        speed_2d_mps, speed_fwd_mps, area_px
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("  Computing background model...")
    background = compute_background(video_path, H=H)
    print("  Detecting swim direction...")
    head_left = _detect_swim_direction(video_path, background, H=H)
    print(f"  Direction: {'left→right' if head_left else 'right→left'}")
    if robot_length_px is None and midline_method == "chain":
        print("  Estimating robot length...")
        robot_length_px = _estimate_robot_length(video_path, background, H=H)

    frames, times, xs_raw, ys_raw, areas = [], [], [], [], []
    annotated_frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H is not None:
            frame = apply_homography(frame, H)

        det = detect_robot(frame, background=background, debug=save_annotated,
                           midline_method=midline_method,
                           robot_length_px=robot_length_px,
                           head_is_left=head_left)
        frames.append(frame_idx)
        times.append(frame_idx / fps)
        if det.head_pt is not None:
            xs_raw.append(float(det.head_pt[0]))
            ys_raw.append(float(det.head_pt[1]))
        else:
            xs_raw.append(np.nan)
            ys_raw.append(np.nan)
        areas.append(det.area)

        if save_annotated and det.detected:
            vis = frame.copy()
            if det.contour is not None:
                overlay = vis.copy()
                cv2.drawContours(overlay, [det.contour], -1, (255, 100, 0), -1)
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
                cv2.drawContours(vis, [det.contour], -1, (255, 180, 0), 2)
            _draw_midline(vis, det)
            cv2.circle(vis, (int(det.cx), int(det.cy)), 8, (0, 255, 0), -1)
            annotated_frames.append(vis)

        frame_idx += 1

    cap.release()

    ts  = np.array(times, dtype=float)
    xs  = np.array(xs_raw, dtype=float)
    ys  = np.array(ys_raw, dtype=float)
    ar  = np.array(areas,  dtype=float)

    # # ── 1. reject glitch jumps ────────────────────────────────────────────
    # xs, ys = _reject_outliers(xs, ys)

    # ── 2. convert to metres ──────────────────────────────────────────────
    xm = xs * meters_per_pixel
    ym = ys * meters_per_pixel

    # ── 3. quadratic polynomial fit: mean swimming path ───────────────────
    if len(ts) < 3:
        sys.exit(f"[ERROR] Too few frames tracked ({len(ts)}). "
                 "Check that the video path is correct and the robot is visible.")
    valid = ~np.isnan(xm)
    if valid.sum() < 3:
        sys.exit("[ERROR] Too few valid detections for polynomial fit.")

    # fit degree-2 polynomial to valid frames only
    px_coeffs = np.polyfit(ts[valid], xm[valid], 2)
    py_coeffs = np.polyfit(ts[valid], ym[valid], 2)

    # evaluate fitted path at every frame time (used for preview + direction)
    xm_fit = np.polyval(px_coeffs, ts)
    ym_fit = np.polyval(py_coeffs, ts)

    # ── 4. low-pass filter positions at 3 Hz (removes undulation noise) ──────
    xm_sm = _lowpass(xm, fps, cutoff_hz=3.0)
    ym_sm = _lowpass(ym, fps, cutoff_hz=3.0)

    # ── 5. differentiate smoothed actual positions ────────────────────────
    dt = np.gradient(ts)
    vx = np.gradient(xm_sm) / dt
    vy = np.gradient(ym_sm) / dt

    speed_2d = np.sqrt(vx**2 + vy**2)

    # ── 6. forward speed: project velocity onto instantaneous quadratic dir ─
    # Direction from quadratic tangent: d/dt [fit] = 2a*t + b
    tx = 2 * px_coeffs[0] * ts + px_coeffs[1]
    ty = 2 * py_coeffs[0] * ts + py_coeffs[1]
    tnorm = np.hypot(tx, ty)
    tnorm = np.where(tnorm > 0, tnorm, 1.0)
    tx_n, ty_n = tx / tnorm, ty / tnorm

    # signed projection: positive = swimming forward along the fitted path
    speed_fwd_raw = vx * tx_n + vy * ty_n
    # low-pass speed signal at 3 Hz to remove residual differentiation noise
    speed_fwd = _lowpass(speed_fwd_raw, fps, cutoff_hz=3.0)
    speed_2d  = _lowpass(speed_2d,      fps, cutoff_hz=3.0)

    df = pd.DataFrame({
        "frame":         frames,
        "time_s":        ts,
        "x_px":          xs_raw,
        "y_px":          ys_raw,
        "x_m":           xm,
        "y_m":           ym,
        "x_fit":         xm_fit,
        "y_fit":         ym_fit,
        "x_sm":          xm_sm,
        "y_sm":          ym_sm,
        "speed_2d_mps":  speed_2d,
        "speed_fwd_mps": speed_fwd,
        "area_px":       ar,
    })

    if save_annotated and annotated_frames:
        out_path = os.path.splitext(video_path)[0] + "_tracked.mp4"
        h_f, w_f = annotated_frames[0].shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w_f, h_f))
        for f in annotated_frames:
            writer.write(f)
        writer.release()
        print(f"  Annotated video → {out_path}")

    return df


def steady_state_speed(df: pd.DataFrame, skip_frac: float = 0.2) -> float:
    """
    Mean forward speed over the steady portion of the recording.

    Uses speed_fwd_mps (projected onto swimming axis) which is free of
    lateral undulation noise.  Skips the first skip_frac of the recording
    (ramp-up) and takes the absolute value before averaging (handles videos
    where the robot swims in the negative direction).
    """
    t0 = df["time_s"].iloc[0]
    t1 = df["time_s"].iloc[-1]
    steady = df[df["time_s"] >= t0 + skip_frac * (t1 - t0)].copy()
    steady["abs_fwd"] = steady["speed_fwd_mps"].abs()
    return steady["abs_fwd"].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_all(summary_csv: str, video_dir: str, output_dir: str,
                meters_per_pixel: float,
                H: np.ndarray | None = None,
                save_annotated: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    summary = pd.read_csv(summary_csv)

    results = []
    for _, row in summary.iterrows():
        vpath = os.path.join(video_dir, row["video_filename"])
        if not os.path.exists(vpath):
            print(f"[SKIP] {vpath} not found")
            continue

        print(f"Processing {row['video_filename']} ...")
        df = track_video(vpath, meters_per_pixel, H=H, save_annotated=save_annotated)

        stem = os.path.splitext(row["video_filename"])[0]
        df.to_csv(os.path.join(output_dir, f"{stem}_track.csv"), index=False)

        mean_spd = steady_state_speed(df)
        print(f"  mean speed = {mean_spd:.4f} m/s")

        results.append({
            "video_filename": row["video_filename"],
            "positionAmplitude_deg": row["positionAmplitude_deg"],
            "Lambda": row["Lambda"],
            "Frequency_Hz": row["Frequency_Hz"],
            "mean_speed_mps": mean_spd,
        })

        # Per-video plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(df["time_s"], df["x_sm"], label="x")
        axes[0].plot(df["time_s"], df["y_sm"], label="y")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend()
        axes[0].set_title(f"{row['video_filename']}  |  "
                          f"A={row['positionAmplitude_deg']}°  "
                          f"f={row['Frequency_Hz']} Hz")
        axes[1].plot(df["time_s"], df["speed_fwd_mps"].abs(), color="tab:orange",
                     label="fwd speed")
        axes[1].axhline(mean_spd, color="red", ls="--",
                        label=f"mean={mean_spd:.3f} m/s")
        axes[1].set_ylabel("Speed (m/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{stem}_speed.png"), dpi=120)
        plt.close(fig)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "speed_summary.csv"), index=False)
    print(f"\nSaved speed_summary.csv with {len(res_df)} entries")
    plot_summary(res_df, output_dir)


def plot_summary(res_df: pd.DataFrame, output_dir: str):
    amplitudes = sorted(res_df["positionAmplitude_deg"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for amp in amplitudes:
        sub = res_df[res_df["positionAmplitude_deg"] == amp]
        grouped = sub.groupby("Frequency_Hz")["mean_speed_mps"].agg(
            ["mean", "std"]).reset_index()
        ax.errorbar(grouped["Frequency_Hz"], grouped["mean"],
                    yerr=grouped["std"], marker="o", capsize=4,
                    label=f"A={amp}°")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean swimming speed (m/s)")
    ax.set_title("Robot swimming speed — experiment")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "speed_vs_frequency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Summary plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(video_path: str, H: np.ndarray | None = None):
    """Click two points with a known real-world distance to get meters_per_pixel."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(5 * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Could not read frame for calibration.")

    if H is not None:
        frame = apply_homography(frame, H)

    points: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow("Calibration", frame)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1280, 720)
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", on_click)

    print("\n[Calibration] Click TWO points with a known real-world distance.")
    print("Press 'q' to finish after clicking both points.\n")
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q") and len(points) == 2:
            break
    cv2.destroyAllWindows()

    px_dist = np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    real_dist = float(
        input(f"Pixel distance = {px_dist:.1f} px. Enter real-world distance in metres: "))
    mpp = real_dist / px_dist
    print(f"\n  → meters_per_pixel = {mpp:.6f}")
    print(f"  Use: --meters_per_pixel {mpp:.6f}\n")
    return mpp


# ─────────────────────────────────────────────────────────────────────────────
# Interactive HSV tuner
# ─────────────────────────────────────────────────────────────────────────────

def tune_hsv(video_path: str, H: np.ndarray | None = None):
    """Slider GUI to tune HSV thresholds.  Press 'n'/'p' to jump frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    def nothing(_): pass

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 1280, 400)
    for name, val in [("H_low", 12), ("H_high", 35), ("S_low", 80),
                       ("S_high", 210), ("V_low", 80), ("V_high", 255)]:
        cv2.createTrackbar(name, "Mask", val, 255, nothing)

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx = 0
            continue

        if H is not None:
            frame = apply_homography(frame, H)

        hl = cv2.getTrackbarPos("H_low",  "Mask")
        hh = cv2.getTrackbarPos("H_high", "Mask")
        sl = cv2.getTrackbarPos("S_low",  "Mask")
        sh = cv2.getTrackbarPos("S_high", "Mask")
        vl = cv2.getTrackbarPos("V_low",  "Mask")
        vh = cv2.getTrackbarPos("V_high", "Mask")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        combined = np.hstack([cv2.resize(frame,  (960, 270)),
                               cv2.resize(masked, (960, 270))])
        cv2.imshow("Mask", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            frame_idx = min(frame_idx + int(fps),
                            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        elif key == ord("p"):
            frame_idx = max(0, frame_idx - int(fps))

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal HSV: lower=({hl},{sl},{vl})  upper=({hh},{sh},{vh})")
    print("Update HSV_YELLOW_LOWER / HSV_YELLOW_UPPER at the top of the script if needed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SUMMARY = os.path.join(
    os.path.dirname(__file__),
    "../../..",          # repo root
    "data/andreaferrario/1guilla_experiments/swim/swim_experiment_summary.csv")
DEFAULT_VIDEO_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../..",
    "data/andreaferrario/1guilla_experiments/swim/videos")
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "../../..",
    "data/andreaferrario/1guilla_experiments/swim/tracking_results")


def plot_track(df: pd.DataFrame, title: str = "") -> None:
    """Plot position and forward speed from a tracking DataFrame."""
    spd = steady_state_speed(df)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(df["time_s"], df["x_sm"], label="x")
    axes[0].plot(df["time_s"], df["y_sm"], label="y")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend()
    if title:
        axes[0].set_title(title)
    axes[1].plot(df["time_s"], df["speed_fwd_mps"].abs(),
                 color="tab:orange", label="fwd speed")
    axes[1].axhline(spd, color="red", ls="--", label=f"mean = {spd:.3f} m/s")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def pick_robot_length(video_path: str, H: np.ndarray | None = None) -> float:
    """
    Show the first frame and ask the user to click head then tail.
    Returns the pixel distance between the two clicks.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Could not read first frame for robot length pick.")
    if H is not None:
        frame = apply_homography(frame, H)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(frame_rgb)
    ax.set_title("Click HEAD then TAIL of the robot  (close window when done)",
                 fontsize=12)
    ax.axis("off")

    clicks: list[tuple[float, float]] = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1 or len(clicks) >= 2:
            return
        clicks.append((event.xdata, event.ydata))
        label = ["HEAD", "TAIL"][len(clicks) - 1]
        ax.plot(event.xdata, event.ydata, 'r+', ms=18, mew=3)
        ax.annotate(label, (event.xdata, event.ydata), color='red', fontsize=11,
                    xytext=(8, 8), textcoords='offset points')
        if len(clicks) == 2:
            dist = np.hypot(clicks[1][0] - clicks[0][0],
                            clicks[1][1] - clicks[0][1])
            ax.plot([clicks[0][0], clicks[1][0]],
                    [clicks[0][1], clicks[1][1]], 'r--', lw=1.5)
            ax.set_title(f"Robot length = {dist:.1f} px  (close to confirm)",
                         fontsize=12)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

    if len(clicks) < 2:
        sys.exit("Need exactly 2 clicks (head and tail).")
    dist = float(np.hypot(clicks[1][0] - clicks[0][0],
                          clicks[1][1] - clicks[0][1]))
    print(f"Robot length: {dist:.1f} px")
    return dist


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument("--meters_per_pixel", type=float, default=None,
                        help="Scale factor in m/px (required unless --calibrate)")

    # Homography
    hom = parser.add_mutually_exclusive_group()
    hom.add_argument("--homography", action="store_true",
                     help="Pick 4 corners interactively to compute homography, "
                          "save as <video>_homography.npy, then track")
    hom.add_argument("--homography_file", metavar="PATH",
                     help="Load pre-computed homography matrix (.npy)")

    # Robot length (chain midline)
    rlen = parser.add_mutually_exclusive_group()
    rlen.add_argument("--robot_length_px", type=float, default=None,
                      help="Robot body length in pixels for chain midline "
                           "(auto-estimated from blob if not given)")
    rlen.add_argument("--pick_length", action="store_true",
                      help="Interactively click head+tail on first frame to "
                           "measure robot length in pixels")

    # Modes
    parser.add_argument("--plot", metavar="CSV",
                        help="Plot speed from an existing _track.csv "
                             "(skips tracking, no other flags needed)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactive scale calibration on --video")
    parser.add_argument("--tune", action="store_true",
                        help="Interactive HSV tuner on --video")
    parser.add_argument("--preview", action="store_true",
                        help="Show live detection debug view with chain midline. "
                             "Requires --video.  SPACE=pause, n/p=step, q=quit")
    parser.add_argument("--save_preview", metavar="PATH",
                        help="Save the debug preview video to this .mp4 path "
                             "(used together with --preview)")
    parser.add_argument("--annotated", action="store_true",
                        help="Save annotated videos with midline overlay")
    parser.add_argument("--midline_method", choices=["chain", "colwise"],
                        default="chain",
                        help="Midline extraction method: "
                             "'chain' (default, MATLAB chain-walking) or "
                             "'colwise' (column-wise top/bottom average)")

    # Batch paths
    parser.add_argument("--summary_csv",
                        default="/data/andreaferrario/1guilla_experiments/swim/"
                                "swim_experiment_summary.csv")
    parser.add_argument("--video_dir",
                        default="/data/andreaferrario/1guilla_experiments/swim/videos")
    parser.add_argument("--output_dir",
                        default="/data/andreaferrario/1guilla_experiments/swim/"
                                "tracking_results")
    args = parser.parse_args()

    # ── resolve homography ────────────────────────────────────────────────────
    H = None
    if args.homography:
        if not args.video:
            parser.error("--homography requires --video")
        H = interactive_homography(args.video)
    elif args.homography_file:
        H = load_homography(args.homography_file)

    # ── resolve robot length ──────────────────────────────────────────────────
    robot_length_px = args.robot_length_px
    if args.pick_length:
        if not args.video:
            parser.error("--pick_length requires --video")
        robot_length_px = pick_robot_length(args.video, H=H)

    # ── mode dispatch ─────────────────────────────────────────────────────────
    if args.plot:
        df = pd.read_csv(args.plot)
        plot_track(df, title=os.path.basename(args.plot))
        return

    if args.tune:
        if not args.video:
            parser.error("--tune requires --video")
        tune_hsv(args.video, H=H)
        return

    if args.preview:
        if not args.video:
            parser.error("--preview requires --video")
        preview_video(args.video, H=H, save_path=args.save_preview,
                      robot_length_px=robot_length_px,
                      midline_method=args.midline_method)
        return

    if args.calibrate:
        if not args.video:
            parser.error("--calibrate requires --video")
        calibrate(args.video, H=H)
        return

    if args.meters_per_pixel is None:
        parser.error("--meters_per_pixel is required (or run --calibrate first)")

    if args.video:
        print(f"Tracking {args.video} ...")
        df = track_video(args.video, args.meters_per_pixel,
                         H=H, save_annotated=args.annotated,
                         robot_length_px=robot_length_px,
                         midline_method=args.midline_method)
        spd = steady_state_speed(df)
        out_csv = os.path.splitext(args.video)[0] + "_track.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Mean speed = {spd:.4f} m/s  |  track → {out_csv}")
        plot_track(df, title=os.path.basename(args.video))
    else:
        process_all(
            summary_csv=args.summary_csv,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            meters_per_pixel=args.meters_per_pixel,
            H=H,
            save_annotated=args.annotated,
        )


if __name__ == "__main__":
    main()
