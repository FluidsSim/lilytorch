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
from scipy.ndimage import median_filter

# Add process_videos to path so we can import video_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "process_videos"))
from video_utils import pick_four_points, pick_sync_frame, compute_homography  # noqa: E402

# ── HSV thresholds for the yellow robot ──────────────────────────────────────
# Robot: H≈17, S≈158, V≈137  |  Pool border: H≈21, S≈230, V≈176
# Capping S at 210 cleanly excludes the saturated yellow pool border.
HSV_LOWER = np.array([12,  80,  80])
HSV_UPPER = np.array([35, 210, 255])

# Water ROI — excludes yellow pool borders (top/bottom strips of the frame).
# Adjust if the camera angle or pool changes.
ROI_Y1, ROI_Y2 = 100, 730

# Morphological kernel to close gaps in the robot mask
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Minimum blob area (px²) to be a valid detection
MIN_AREA_PX = 500

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
# Core detection
# ─────────────────────────────────────────────────────────────────────────────

_roi_mask_cache: dict = {}


def detect_robot(frame: np.ndarray) -> tuple[float | None, float | None, float]:
    """Return (cx, cy, area) of the largest yellow blob, or (None, None, 0)."""
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in _roi_mask_cache:
        m = np.zeros((h, w), np.uint8)
        m[ROI_Y1:min(ROI_Y2, h), :] = 255
        _roi_mask_cache[key] = m
    roi = _roi_mask_cache[key]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_AREA_PX:
        return None, None, area

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, None, area
    return M["m10"] / M["m00"], M["m01"] / M["m00"], area


# ─────────────────────────────────────────────────────────────────────────────
# Track a single video
# ─────────────────────────────────────────────────────────────────────────────

def track_video(video_path: str, meters_per_pixel: float,
                H: np.ndarray | None = None,
                save_annotated: bool = False) -> pd.DataFrame:
    """
    Track the robot centroid through every frame.

    If H is provided, each frame is perspective-corrected before detection.

    Returns a DataFrame with columns:
        frame, time_s, x_px, y_px, x_m, y_m, x_m_sm, y_m_sm, speed_mps, area_px
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames, times, xs, ys, areas = [], [], [], [], []
    annotated_frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if H is not None:
            frame = apply_homography(frame, H)

        cx, cy, area = detect_robot(frame)
        frames.append(frame_idx)
        times.append(frame_idx / fps)
        xs.append(cx)
        ys.append(cy)
        areas.append(area)

        if save_annotated and cx is not None:
            vis = frame.copy()
            cv2.circle(vis, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            annotated_frames.append(vis)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame({
        "frame": frames,
        "time_s": times,
        "x_px": xs,
        "y_px": ys,
        "area_px": areas,
    })
    df = df.dropna(subset=["x_px", "y_px"]).copy()

    df["x_m"] = df["x_px"] * meters_per_pixel
    df["y_m"] = df["y_px"] * meters_per_pixel

    df["x_m_sm"] = median_filter(df["x_m"].values, size=5)
    df["y_m_sm"] = median_filter(df["y_m"].values, size=5)

    dt = np.gradient(df["time_s"].values)
    dx = np.gradient(df["x_m_sm"].values)
    dy = np.gradient(df["y_m_sm"].values)
    df["speed_mps"] = median_filter(np.sqrt(dx**2 + dy**2) / dt, size=9)

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
    """Mean speed over the steady portion (skip early ramp-up + outliers)."""
    t0 = df["time_s"].iloc[0]
    t1 = df["time_s"].iloc[-1]
    steady = df[df["time_s"] >= t0 + skip_frac * (t1 - t0)]
    med = steady["speed_mps"].median()
    steady = steady[steady["speed_mps"] < 3 * med]
    return steady["speed_mps"].mean()


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
        axes[0].plot(df["time_s"], df["x_m_sm"], label="x")
        axes[0].plot(df["time_s"], df["y_m_sm"], label="y")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend()
        axes[0].set_title(f"{row['video_filename']}  |  "
                          f"A={row['positionAmplitude_deg']}°  "
                          f"f={row['Frequency_Hz']} Hz")
        axes[1].plot(df["time_s"], df["speed_mps"], color="tab:orange")
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
    print("Update HSV_LOWER / HSV_UPPER at the top of the script if needed.\n")


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

    # Modes
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactive scale calibration on --video")
    parser.add_argument("--tune", action="store_true",
                        help="Interactive HSV tuner on --video")
    parser.add_argument("--annotated", action="store_true",
                        help="Save annotated videos with centroid overlay")

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

    # ── mode dispatch ─────────────────────────────────────────────────────────
    if args.tune:
        if not args.video:
            parser.error("--tune requires --video")
        tune_hsv(args.video, H=H)
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
                         H=H, save_annotated=args.annotated)
        spd = steady_state_speed(df)
        out_csv = os.path.splitext(args.video)[0] + "_track.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Mean speed = {spd:.4f} m/s  |  track → {out_csv}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].plot(df["time_s"], df["x_m_sm"], label="x")
        axes[0].plot(df["time_s"], df["y_m_sm"], label="y")
        axes[0].set_ylabel("Position (m)")
        axes[0].legend()
        axes[1].plot(df["time_s"], df["speed_mps"], color="tab:orange")
        axes[1].axhline(spd, color="red", ls="--", label=f"mean={spd:.3f} m/s")
        axes[1].set_ylabel("Speed (m/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        fig.tight_layout()
        plt.show()
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
