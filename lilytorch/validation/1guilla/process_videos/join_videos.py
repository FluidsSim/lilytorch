"""
Join a corrected experimental video and a simulation video into a
vertically-stacked comparison.

Usage
-----
    python join_videos.py <corrected_video1> <video2>

    corrected_video1 — output of correct_video1.py (already trimmed +
                       perspective-corrected).
    video2           — simulation/reference video (raw).

Steps
-----
  1. Temporal sync — scrub video2 to its motion-start and motion-end frames.
     (video1 is assumed to already start at motion onset.)
  2. Metric scale  — click 2 points of known distance in each video.
     video2 is rescaled so 1 m spans the same pixels as video1.
  3. Spatial anchor — click one common landmark in both videos.
     video2 is shifted so the landmark aligns pixel-for-pixel.
  4. Encoding      — both videos are resampled to a common output fps (30)
     and written as intermediate files.
  5. ffmpeg stack  — top = experiment, bottom = simulation, 1920 px wide.

Output
------
    stacked_output.mp4  (in the same directory as video1)
"""

import argparse
import cv2
import numpy as np
import subprocess
import sys
import os


# ── parse arguments ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Join corrected experiment video and simulation video.")
parser.add_argument("video1",
                    help="Corrected experiment video (output of correct_video1.py)")
parser.add_argument("video2",
                    help="Simulation / reference video (raw)")
args = parser.parse_args()

INPUT1 = os.path.abspath(args.video1)
INPUT2 = os.path.abspath(args.video2)
DIR    = os.path.dirname(INPUT1)

base2, ext2 = os.path.splitext(os.path.basename(INPUT2))
OUT1_SYNCED = os.path.join(
    DIR, os.path.splitext(os.path.basename(INPUT1))[0] + "_synced"
    + os.path.splitext(INPUT1)[1])
OUT2 = os.path.join(DIR, base2 + "_corrected" + ext2)
STACKED = os.path.join(DIR, "stacked_output.mp4")

for p in (INPUT1, INPUT2):
    if not os.path.exists(p):
        sys.exit(f"File not found: {p}")

# Read native frame rates
_c = cv2.VideoCapture(INPUT1); FPS1 = _c.get(cv2.CAP_PROP_FPS); _c.release()
_c = cv2.VideoCapture(INPUT2); FPS2 = _c.get(cv2.CAP_PROP_FPS); _c.release()
OUTPUT_FPS = 30.0

print(f"video1 : {INPUT1}")
print(f"video2 : {INPUT2}")
print(f"  video1 fps: {FPS1:.2f}  |  video2 fps: {FPS2:.2f}  |  output fps: {OUTPUT_FPS}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def pick_sync_frame(path, title):
    """Scrubber window.  ENTER selects the current frame index."""
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
            cv2.putText(vis,
                        f"Frame {idx}/{total-1}  |  ENTER=select  r=rewind  ESC=cancel",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(title, vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            cap.release(); cv2.destroyAllWindows(); return idx
        elif key == ord('r'):
            state["idx"] = 0; cv2.setTrackbarPos("Frame", title, 0)
        elif key == 27:
            cap.release(); cv2.destroyAllWindows(); sys.exit("Cancelled")


def pick_two_points(frame, title):
    """Click 2 points.  Returns list of 2 (x, y)."""
    pts = []

    def draw(img, pts):
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 8, (0, 255, 0), -1)
            cv2.putText(vis, f"P{i+1}", (p[0]+10, p[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], (0, 255, 255), 2)
            d = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            cv2.putText(vis, f"{d:.1f} px",
                        ((pts[0][0]+pts[1][0])//2+10,
                         (pts[0][1]+pts[1][1])//2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        msg = (f"Click P{len(pts)+1}" if len(pts) < 2
               else "Press ENTER to confirm, 'r' to reset")
        cv2.putText(vis, msg, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
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


def pick_one_point(frame, title):
    """Click 1 anchor point (re-clickable).  Returns (x, y)."""
    pts = []

    def draw(img, pts):
        vis = img.copy()
        if pts:
            cv2.circle(vis, pts[0], 10, (0, 0, 255), -1)
            cv2.putText(vis, "Anchor", (pts[0][0]+10, pts[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        msg = ("Click the anchor point" if not pts
               else "Press ENTER to confirm, 'r' to reset")
        cv2.putText(vis, msg, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
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


def crop_with_anchor(frame, target_w, target_h, anchor_src, anchor_dst):
    """Crop/pad so that anchor_src lands on anchor_dst in the output."""
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
# STEP 1 — Temporal sync for video2
#   video1 is already trimmed by correct_video1.py — frame 0 = motion start.
#   We only need to pick start & end for video2.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Temporal sync for video2")
print("=" * 60)
print("(video1 is already trimmed — frame 0 = motion start.)")

print("\nScrub video2 to the frame where the object STARTS moving.")
sync2_start = pick_sync_frame(INPUT2, "video2 — motion START frame")
print(f"  video2 start frame: {sync2_start}")

print("\nScrub video2 to the frame where you want it to END.")
sync2_end = pick_sync_frame(INPUT2, "video2 — END frame")
print(f"  video2 end frame:   {sync2_end}")

if sync2_end <= sync2_start:
    sys.exit(f"video2 end frame ({sync2_end}) must be after start ({sync2_start})")

# video1 total frame count (already trimmed)
_c = cv2.VideoCapture(INPUT1); total1 = int(_c.get(cv2.CAP_PROP_FRAME_COUNT)); _c.release()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Metric scale matching
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 2 — Metric scale matching")
print("=" * 60)

# Read reference frames
frame1 = get_first_frame(INPUT1)  # first frame of corrected video1
cap2 = cv2.VideoCapture(INPUT2)
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2_start)
ret2, frame2 = cap2.read()
cap2.release()
if not ret2:
    sys.exit(f"Cannot read frame {sync2_start} from {INPUT2}")

ppm1 = pixel_per_metre(frame1,
                        "video1 (corrected) — click 2 points of known distance")
ppm2 = pixel_per_metre(frame2,
                        "video2 (motion start) — click 2 points of known distance")

scale_factor = ppm1 / ppm2
H1, W1 = frame1.shape[:2]
H2, W2 = frame2.shape[:2]
new_w = int(round(W2 * scale_factor))
new_h = int(round(H2 * scale_factor))
print(f"\n  scale factor: {scale_factor:.4f}")
print(f"  video2 {W2}×{H2} → rescaled {new_w}×{new_h} → output {W1}×{H1}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Spatial anchor alignment
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 3 — Spatial anchor alignment")
print("=" * 60)
print("Click ONE fixed landmark visible in both videos (e.g. tank corner).")

anchor1 = pick_one_point(frame1, "video1 — click the anchor landmark")
print(f"  video1 anchor: {anchor1}")

frame2_preview = cv2.resize(frame2, (new_w, new_h),
                             interpolation=cv2.INTER_LANCZOS4)
anchor2 = pick_one_point(
    frame2_preview,
    f"video2 (rescaled {new_w}×{new_h}) — click the SAME landmark")
print(f"  video2 anchor (rescaled): {anchor2}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Encode both synced output videos
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 4 — Encoding synced videos")
print("=" * 60)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# ── video1: resample to OUTPUT_FPS ────────────────────────────────────────────
n1_src = total1
duration1 = n1_src / FPS1
n1_out = int(round(duration1 * OUTPUT_FPS))

cap1s = cv2.VideoCapture(INPUT1)
writer1s = cv2.VideoWriter(OUT1_SYNCED, fourcc, OUTPUT_FPS, (W1, H1))
print(f"video1: {n1_src} src frames ({FPS1:.1f} fps) → {n1_out} out frames "
      f"({OUTPUT_FPS:.0f} fps)  [{duration1:.2f} s]")
print(f"  → {OUT1_SYNCED}")

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
while out_written < n1_out and last_frame is not None:
    writer1s.write(last_frame)
    out_written += 1
cap1s.release()
writer1s.release()
print(f"\n  ✓ Saved {OUT1_SYNCED}")

# ── video2: rescale + anchor + resample to OUTPUT_FPS ─────────────────────────
n2_src = sync2_end - sync2_start
duration2 = n2_src / FPS2
n2_out = int(round(duration2 * OUTPUT_FPS))

cap2 = cv2.VideoCapture(INPUT2)
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2_start)
writer2 = cv2.VideoWriter(OUT2, fourcc, OUTPUT_FPS, (W1, H1))
print(f"\nvideo2: {n2_src} src frames ({FPS2:.1f} fps) → {n2_out} out frames "
      f"({OUTPUT_FPS:.0f} fps)  [{duration2:.2f} s]")
print(f"  → {OUT2}")

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
while out_written < n2_out and last_frame is not None:
    writer2.write(last_frame)
    out_written += 1
cap2.release()
writer2.release()
print(f"\n  ✓ Saved {OUT2}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Stack with ffmpeg
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 5 — Stacking with ffmpeg")
print("=" * 60)

cmd = [
    "ffmpeg", "-y",
    "-i", OUT1_SYNCED,
    "-i", OUT2,
    "-filter_complex",
    "[0:v]scale=1920:-2[top];[1:v]scale=1920:-2[bot];"
    "[top][bot]vstack=inputs=2[out]",
    "-map", "[out]",
    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
    STACKED,
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd)
if result.returncode != 0:
    sys.exit("ffmpeg failed — check output above.")

print()
print("=" * 60)
print(f"Done!  {STACKED}")
print("=" * 60)
