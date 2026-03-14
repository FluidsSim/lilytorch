"""
Experiment-vs-simulation video comparison pipeline.

Creates a vertically-stacked comparison video between an experimental
recording (e.g. GoPro footage of a robot in a water tank) and a CFD
simulation rendering.  The pipeline performs perspective correction,
temporal synchronisation, metric scale matching, and spatial anchoring
so that the two videos are pixel-aligned and start at the same instant.

Usage
-----
    python pipeline.py <video1> <video2> [--flip]

    video1  – experimental recording (e.g. GoPro .MP4).  Will be perspective-
              corrected (and optionally horizontally flipped with --flip).
    video2  – simulation/reference video (.mp4).  Will be scaled, spatially
              anchored, and temporally synced to video1.
    --flip  – horizontally flip video1 after perspective correction.

Pipeline steps
--------------
  Step 1 – Temporal synchronisation
      Scrub each **raw** video to the frame where motion begins (start
      frame) and to the frame where it ends (end frame).  The start
      frame is reused for all subsequent interactive picking steps.

  Step 2 – Perspective correction (+ optional flip) for video1
      Uses the start frame chosen in Step 1 for 4-corner picking
      (the relevant features are more visible at motion onset).
      Click 4 corners of a known rectangle → homography warp.
      If --flip is passed, also applies a horizontal mirror.
      Output: <video1>_corrected.<ext>

  Step 3 – Metric scale matching
      In each video, click 2 points of known real-world distance and type
      the distance in metres (using the motion-start frame).  video2 is
      rescaled so 1 m occupies the same number of pixels as in video1.

  Step 4 – Spatial anchor alignment
      Click one fixed landmark visible in both videos (e.g. tank corner).
      video2 is cropped/padded so the landmark sits at the same pixel.

  Step 5 – Encoding
      Writes <video1>_synced.<ext> and <video2>_corrected.<ext>, both
      trimmed to [start, end], with matched resolution.

  Step 6 – Vertical stacking (ffmpeg)
      Top = experiment, bottom = simulation, both 1920 px wide.
      Output: stacked_output.mp4  (H.264, CRF 18)

Output files (written next to the input videos)
------------------------------------------------
    <video1>_corrected.<ext>   – perspective-corrected (+ flipped) video1
    <video1>_synced.<ext>      – corrected video1 trimmed to [start, end]
    <video2>_corrected.<ext>   – scaled + anchored + synced video2
    stacked_output.mp4         – final top/bottom composite
    video_sync.env             – video1 start offset (for stack_videos.sh compat)

Standalone scripts
------------------
  Each step can also be run independently:
    correct_video1.py      – step 1 only
    match_scale_video2.py  – steps 2-4 + video2 encode
    stack_videos.sh        – ffmpeg vertical stack

Requirements
------------
  Python 3, opencv-python, numpy, ffmpeg on PATH, a display for the
  interactive OpenCV windows.

Not referenced elsewhere in the codebase – this is a standalone CLI tool.
"""

import argparse
import cv2
import numpy as np
import subprocess
import sys
import os


# ── parse arguments ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Experiment-vs-simulation video comparison pipeline.")
parser.add_argument("video1", help="Experimental recording (e.g. GoPro .MP4)")
parser.add_argument("video2", help="Simulation/reference video (.mp4)")
parser.add_argument("--flip", action="store_true",
                    help="Horizontally flip video1 after perspective correction")
parser.add_argument("--skip-correction", action="store_true",
                    help="Skip Step 2 (perspective correction) and reuse the existing *_corrected file")
args = parser.parse_args()

INPUT1  = os.path.abspath(args.video1)
INPUT2  = os.path.abspath(args.video2)
FLIP    = args.flip
SKIP_CORRECTION = args.skip_correction
DIR     = os.path.dirname(INPUT1)

# Helper to append '_corrected' before extension
def corrected_name(path):
    base, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(DIR, base + '_corrected' + ext)

OUT1 = corrected_name(INPUT1)
OUT2 = corrected_name(INPUT2)
OUT1_SYNCED = os.path.join(DIR, os.path.splitext(os.path.basename(INPUT1))[0] + '_synced' + os.path.splitext(INPUT1)[1])
STACKED = os.path.join(DIR, "stacked_output.mp4")
ENVFILE = os.path.join(DIR, "video_sync.env")

for p in (INPUT1, INPUT2):
    if not os.path.exists(p):
        sys.exit(f"File not found: {p}")

print(f"video1 : {INPUT1}")
print(f"video2 : {INPUT2}")

# Read native frame rates
_c = cv2.VideoCapture(INPUT1); FPS1 = _c.get(cv2.CAP_PROP_FPS); _c.release()
_c = cv2.VideoCapture(INPUT2); FPS2 = _c.get(cv2.CAP_PROP_FPS); _c.release()
OUTPUT_FPS = 30.0  # common output frame rate
print(f"  video1 fps: {FPS1:.2f}  |  video2 fps: {FPS2:.2f}  |  output fps: {OUTPUT_FPS}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_first_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Cannot read first frame from {path}")
    return frame


def show_window(title, frame, w=1280, h=720):
    h0, w0 = frame.shape[:2]
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(w0, w), min(h0, h))


# ── 4-point picker ────────────────────────────────────────────────────────────

def pick_four_points(frame, title):
    """Click 4 corners of a rectangle. Returns list of 4 (x,y)."""
    LABELS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    COLORS = [(0,255,0), (0,255,255), (0,0,255), (255,0,0)]
    pts = []

    def draw(img, pts):
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 8, COLORS[i], -1)
            cv2.putText(vis, LABELS[i], (p[0]+10, p[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[i], 2)
        if len(pts) == 4:
            cv2.polylines(vis, [np.array(pts)], True, (255,255,0), 2)
        msg = f"Click: {LABELS[len(pts)]}" if len(pts) < 4 else "Press ENTER to confirm, 'r' to reset"
        cv2.putText(vis, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return vis

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.imshow(title, draw(frame, pts))

    show_window(title, frame)
    cv2.setMouseCallback(title, mouse_cb)
    cv2.imshow(title, draw(frame, pts))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            pts.clear(); cv2.imshow(title, draw(frame, pts))
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows(); sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts


# ── 2-point distance picker ───────────────────────────────────────────────────

def pick_two_points(frame, title):
    """Click 2 points. Returns list of 2 (x,y)."""
    pts = []

    def draw(img, pts):
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 8, (0,255,0), -1)
            cv2.putText(vis, f"P{i+1}", (p[0]+10, p[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], (0,255,255), 2)
            d = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            cv2.putText(vis, f"{d:.1f} px",
                        ((pts[0][0]+pts[1][0])//2+10, (pts[0][1]+pts[1][1])//2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        msg = f"Click P{len(pts)+1}" if len(pts) < 2 else "Press ENTER to confirm, 'r' to reset"
        cv2.putText(vis, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return vis

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.imshow(title, draw(frame, pts))

    show_window(title, frame)
    cv2.setMouseCallback(title, mouse_cb)
    cv2.imshow(title, draw(frame, pts))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            pts.clear(); cv2.imshow(title, draw(frame, pts))
        elif key in (13, 10) and len(pts) == 2:
            break
        elif key == 27:
            cv2.destroyAllWindows(); sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts


def pixel_per_metre(frame, title):
    print(f"\n=== {title} ===")
    print("Click 2 points along a known real-world distance.")
    pts = pick_two_points(frame, title)
    px_dist = float(np.linalg.norm(np.array(pts[0]) - np.array(pts[1])))
    while True:
        try:
            m_dist = float(input("  Distance between the two points in metres: "))
            if m_dist > 0:
                break
        except ValueError:
            pass
    ppm = px_dist / m_dist
    print(f"  → {px_dist:.1f} px = {m_dist} m  →  {ppm:.2f} px/m")
    return ppm


# ── 1-point anchor picker ─────────────────────────────────────────────────────

def pick_one_point(frame, title):
    """Click 1 anchor point (re-clickable). Returns (x, y)."""
    pts = []

    def draw(img, pts):
        vis = img.copy()
        if pts:
            cv2.circle(vis, pts[0], 10, (0,0,255), -1)
            cv2.putText(vis, "Anchor", (pts[0][0]+10, pts[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        msg = "Click the anchor point" if not pts else "Press ENTER to confirm, 'r' to reset"
        cv2.putText(vis, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return vis

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.clear(); pts.append((x, y))
            cv2.imshow(title, draw(frame, pts))

    show_window(title, frame)
    cv2.setMouseCallback(title, mouse_cb)
    cv2.imshow(title, draw(frame, pts))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            pts.clear(); cv2.imshow(title, draw(frame, pts))
        elif key in (13, 10) and pts:
            break
        elif key == 27:
            cv2.destroyAllWindows(); sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts[0]


# ── sync-frame scrubber ───────────────────────────────────────────────────────

def pick_sync_frame(path, title):
    """Scrubber window. ENTER selects current frame index."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state = {"idx": 0, "frame": None}

    def on_trackbar(val):
        state["idx"] = val

    h0, w0 = get_first_frame(path).shape[:2]
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(w0, 1280), min(h0, 720))
    cv2.createTrackbar("Frame", title, 0, total - 1, on_trackbar)

    last_idx = -1
    while True:
        idx = state["idx"]
        if idx != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                state["frame"] = frame.copy()
            last_idx = idx
        if state["frame"] is not None:
            vis = state["frame"].copy()
            cv2.putText(vis, f"Frame {idx}/{total-1}  |  ENTER=select  r=rewind  ESC=cancel",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(title, vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            cap.release(); cv2.destroyAllWindows(); return idx
        elif key == ord('r'):
            state["idx"] = 0; cv2.setTrackbarPos("Frame", title, 0)
        elif key == 27:
            cap.release(); cv2.destroyAllWindows(); sys.exit("Cancelled")


# ── anchor-aware crop/pad ─────────────────────────────────────────────────────

def crop_with_anchor(frame, target_w, target_h, anchor_src, anchor_dst):
    h, w = frame.shape[:2]
    offset_x = anchor_src[0] - anchor_dst[0]
    offset_y = anchor_src[1] - anchor_dst[1]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    sx0 = max(0, offset_x);  sy0 = max(0, offset_y)
    sx1 = min(w, offset_x + target_w)
    sy1 = min(h, offset_y + target_h)
    dx0 = max(0, -offset_x); dy0 = max(0, -offset_y)
    dx1 = dx0 + (sx1 - sx0);  dy1 = dy0 + (sy1 - sy0)
    if sx1 > sx0 and sy1 > sy0:
        canvas[dy0:dy1, dx0:dx1] = frame[sy0:sy1, sx0:sx1]
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Temporal sync (on raw videos)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 – Temporal sync (start + end frames)")
print("=" * 60)

print("Scrub each video to the frame where the object STARTS moving, press ENTER.")
sync1_start = pick_sync_frame(INPUT1, "video1 (raw) – motion START frame")
sync2_start = pick_sync_frame(INPUT2, "video2 – motion START frame")
print(f"  video1 start frame: {sync1_start}  |  video2 start frame: {sync2_start}")

print("\nNow scrub each video to the frame where you want the video to END, press ENTER.")
sync1_end = pick_sync_frame(INPUT1, "video1 (raw) – END frame")
sync2_end = pick_sync_frame(INPUT2, "video2 – END frame")
print(f"  video1 end frame: {sync1_end}  |  video2 end frame: {sync2_end}")

if sync1_end <= sync1_start:
    sys.exit(f"video1 end frame ({sync1_end}) must be after start frame ({sync1_start})")
if sync2_end <= sync2_start:
    sys.exit(f"video2 end frame ({sync2_end}) must be after start frame ({sync2_start})")

# Save for stack_videos.sh backward-compat (not used by this pipeline).
with open(ENVFILE, "w") as f:
    f.write(f"VIDEO1_START={sync1_start / FPS1:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Perspective correction (+ optional flip) for video1
# Uses the start frame from Step 1 for corner picking.
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print(f"STEP 2 – Perspective correction{' + flip' if FLIP else ''} for video1")
print("=" * 60)

if SKIP_CORRECTION:
    if not os.path.exists(OUT1):
        sys.exit(f"--skip-correction: {OUT1} does not exist. Run without --skip-correction first.")
    print(f"  Skipping (reusing {OUT1})")
    # Read FPS from the corrected file
    _c = cv2.VideoCapture(OUT1); FPS1 = _c.get(cv2.CAP_PROP_FPS); _c.release()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
else:
    # Read the start frame (easier to pick corners when the object is visible).
    cap_pick = cv2.VideoCapture(INPUT1)
    cap_pick.set(cv2.CAP_PROP_POS_FRAMES, sync1_start)
    ret_pick, frame1_raw = cap_pick.read()
    H_px, W_px = frame1_raw.shape[:2]
    cap_pick.release()
    if not ret_pick:
        sys.exit(f"Cannot read frame {sync1_start} from {INPUT1}")

    print(f"Showing video1 frame {sync1_start} for corner picking.")
    print("Click the 4 corners of a rectangular feature.")
    print("Order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")
    corners = pick_four_points(frame1_raw, "video1 – click 4 rectangle corners")

    src_pts = np.float32(corners)
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dst_pts = np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    H_mat, _ = cv2.findHomography(src_pts, dst_pts)

    # Compute the canvas that contains the ENTIRE warped image (not just the
    # corrected rectangle) so nothing is cropped.
    img_corners = np.float32([[0, 0], [W_px, 0], [W_px, H_px], [0, H_px]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(img_corners, H_mat).reshape(-1, 2)
    all_x = warped_corners[:, 0]; all_y = warped_corners[:, 1]
    min_x, max_x = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    min_y, max_y = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))
    # Translation matrix to shift everything into positive coordinates
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float64)
    H_full = T @ H_mat
    out_w = max_x - min_x
    out_h = max_y - min_y
    print(f"  Perspective warp: {W_px}×{H_px}  →  {out_w}×{out_h}  (full canvas, no cropping)")

    cap1 = cv2.VideoCapture(INPUT1)
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT1, fourcc, FPS1, (out_w, out_h))

    print(f"Encoding {total1} frames → {OUT1}")
    for i in range(total1):
        ret, frame = cap1.read()
        if not ret:
            break
        warped  = cv2.warpPerspective(frame, H_full, (out_w, out_h))
        out_frame = cv2.flip(warped, 1) if FLIP else warped
        writer.write(out_frame)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{total1}", end="\r", flush=True)

    cap1.release()
    writer.release()
    print(f"\n  ✓ Saved {OUT1}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Metric scale matching
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 3 – Metric scale matching")
print("=" * 60)

# Read the corrected start frame from OUT1 (perspective-corrected video1)
# and the raw start frame from INPUT2.
cap1 = cv2.VideoCapture(OUT1)
cap1.set(cv2.CAP_PROP_POS_FRAMES, sync1_start)
ret1, frame1 = cap1.read()
cap1.release()
cap2 = cv2.VideoCapture(INPUT2)
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2_start)
ret2, frame2 = cap2.read()
cap2.release()
if not ret1 or not ret2:
    sys.exit("Could not read frames at sync points for metric scale.")

ppm1 = pixel_per_metre(frame1, "video1 (corrected, motion start) – click 2 points of known distance")
ppm2 = pixel_per_metre(frame2, "video2 (motion start) – click 2 points of known distance")

scale_factor = ppm1 / ppm2
H1, W1 = frame1.shape[:2]
H2, W2 = frame2.shape[:2]
new_w = int(round(W2 * scale_factor))
new_h = int(round(H2 * scale_factor))
print(f"\n  scale factor: {scale_factor:.4f}")
print(f"  video2 {W2}×{H2}  →  rescaled {new_w}×{new_h}  →  output {W1}×{H1}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Spatial anchor
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 4 – Spatial anchor alignment")
print("=" * 60)
print("Click ONE fixed landmark visible in both videos (e.g. a tank corner).")

anchor1 = pick_one_point(frame1, "video1 – click the anchor landmark")
print(f"  video1 anchor: {anchor1}")

frame2_preview = cv2.resize(frame2, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
anchor2 = pick_one_point(frame2_preview,
                         f"video2 (rescaled {new_w}×{new_h}) – click the SAME landmark")
print(f"  video2 anchor (rescaled space): {anchor2}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Encode both synced output videos
# Both are trimmed from their sync frames so frame 0 = motion start.
# No ffmpeg -ss needed → frame-perfect alignment regardless of container timing.
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 5 – Encoding synced video1 and scaled video2")
print("=" * 60)

# ── video1: corrected + trimmed from sync1 ────────────────────────────────────
cap1s = cv2.VideoCapture(OUT1)
cap1s.set(cv2.CAP_PROP_POS_FRAMES, sync1_start)
n1_src = sync1_end - sync1_start
duration1 = n1_src / FPS1
n1_out = int(round(duration1 * OUTPUT_FPS))
writer1s = cv2.VideoWriter(OUT1_SYNCED, fourcc, OUTPUT_FPS, (W1, H1))
print(f"Encoding {n1_src} src frames ({FPS1:.1f} fps) → {n1_out} out frames ({OUTPUT_FPS:.0f} fps)  [{duration1:.2f} s]")
print(f"  video1 frames {sync1_start}–{sync1_end} → {OUT1_SYNCED}")

out_written = 0
last_frame = None
for src_i in range(n1_src):
    ret, frame = cap1s.read()
    if not ret:
        break
    last_frame = frame
    while out_written < n1_out:
        nearest = min(int(out_written * FPS1 / OUTPUT_FPS), n1_src - 1)
        if nearest <= src_i:
            writer1s.write(frame)
            out_written += 1
        else:
            break
    if (src_i + 1) % 100 == 0:
        print(f"  {out_written}/{n1_out}", end="\r", flush=True)
# flush remaining output frames (rounding / upsampling)
while out_written < n1_out and last_frame is not None:
    writer1s.write(last_frame)
    out_written += 1
cap1s.release()
writer1s.release()
print(f"\n  ✓ Saved {OUT1_SYNCED}")

# ── video2: rescaled + anchor-cropped + trimmed from sync2 ────────────────────
cap2 = cv2.VideoCapture(INPUT2)
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2_start)
n2_src = sync2_end - sync2_start
duration2 = n2_src / FPS2
n2_out = int(round(duration2 * OUTPUT_FPS))
writer2 = cv2.VideoWriter(OUT2, fourcc, OUTPUT_FPS, (W1, H1))
print(f"Encoding {n2_src} src frames ({FPS2:.1f} fps) → {n2_out} out frames ({OUTPUT_FPS:.0f} fps)  [{duration2:.2f} s]")
print(f"  video2 frames {sync2_start}–{sync2_end} → {OUT2}")

out_written = 0
last_frame = None
for src_i in range(n2_src):
    ret, frame = cap2.read()
    if not ret:
        break
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    final   = crop_with_anchor(resized, W1, H1, anchor2, anchor1)
    last_frame = final
    while out_written < n2_out:
        nearest = min(int(out_written * FPS2 / OUTPUT_FPS), n2_src - 1)
        if nearest <= src_i:
            writer2.write(final)
            out_written += 1
        else:
            break
    if (src_i + 1) % 50 == 0:
        print(f"  {out_written}/{n2_out}", end="\r", flush=True)
# flush remaining output frames (rounding / upsampling)
while out_written < n2_out and last_frame is not None:
    writer2.write(last_frame)
    out_written += 1
cap2.release()
writer2.release()
print(f"\n  ✓ Saved {OUT2}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Stack with ffmpeg
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 6 – Stacking with ffmpeg")
print("=" * 60)

# Both inputs start at frame 0 = motion start. No -ss needed.
cmd = [
    "ffmpeg", "-y",
    "-i", OUT1_SYNCED,
    "-i", OUT2,
    "-filter_complex",
    "[0:v]scale=1920:-2[top];[1:v]scale=1920:-2[bot];[top][bot]vstack=inputs=2[out]",
    "-map", "[out]",
    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
    STACKED,
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd)
if result.returncode != 0:
    sys.exit("ffmpeg failed – check output above.")

print()
print("=" * 60)
print(f"Pipeline complete!")
print(f"  {STACKED}")
print("=" * 60)
