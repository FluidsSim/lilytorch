"""
Full video comparison pipeline.

Usage
-----
    python pipeline.py <video1> <video2>

    video1  – experimental recording (e.g. GoPro .MP4).  Will be perspective-
              corrected and horizontally flipped.
    video2  – simulation/reference video (.mp4).  Will be scaled, spatially
              anchored, and temporally synced to video1.

Output files (written next to the two input videos):
    video1_corrected.mp4   – corrected video1
    video2_scaled.mp4      – scaled + anchored + synced video2
    stacked_output.mp4     – final top/bottom composite

Interactive steps
-----------------
  1. video1 – click 4 corners of a rectangular feature (perspective fix).
  2. video1 – click 2 points of known distance (m), enter the distance.
  3. video2 – click 2 points of known distance (m), enter the distance.
  4. video1 – click 1 anchor landmark (fixed point visible in both videos).
  5. video2 – click the same anchor landmark (in the rescaled preview).
  6. video1 – scrub to the frame where the object starts moving, press ENTER.
  7. video2 – scrub to the frame where the object starts moving, press ENTER.
"""

import cv2
import numpy as np
import subprocess
import sys
import os


# ── parse arguments ───────────────────────────────────────────────────────────

if len(sys.argv) != 3:
    sys.exit("Usage: python pipeline.py <video1> <video2>")

INPUT1  = os.path.abspath(sys.argv[1])
INPUT2  = os.path.abspath(sys.argv[2])
DIR     = os.path.dirname(INPUT1)
OUT1    = os.path.join(DIR, "video1_corrected.mp4")
OUT1_SYNCED = os.path.join(DIR, "video1_synced.mp4")
OUT2    = os.path.join(DIR, "video2_scaled.mp4")
STACKED = os.path.join(DIR, "stacked_output.mp4")
ENVFILE = os.path.join(DIR, "video_sync.env")

for p in (INPUT1, INPUT2):
    if not os.path.exists(p):
        sys.exit(f"File not found: {p}")

print(f"video1 : {INPUT1}")
print(f"video2 : {INPUT2}")
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
# STEP 1 – Perspective correction + flip → video1_corrected.mp4
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 – Perspective correction + flip for video1")
print("=" * 60)

frame1_raw = get_first_frame(INPUT1)
H_px, W_px = frame1_raw.shape[:2]

print("Click the 4 corners of a rectangular feature.")
print("Order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")
corners = pick_four_points(frame1_raw, "video1 – click 4 rectangle corners")

src_pts = np.float32(corners)
xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
dst_pts = np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
H_mat, _ = cv2.findHomography(src_pts, dst_pts)

cap1 = cv2.VideoCapture(INPUT1)
total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT1, fourcc, 30, (W_px, H_px))

print(f"Encoding {total1} frames → {OUT1}")
for i in range(total1):
    ret, frame = cap1.read()
    if not ret:
        break
    warped  = cv2.warpPerspective(frame, H_mat, (W_px, H_px))
    flipped = cv2.flip(warped, 1)
    writer.write(flipped)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{total1}", end="\r", flush=True)

cap1.release()
writer.release()
print(f"\n  ✓ Saved {OUT1}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Scale measurement
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 2 – Metric scale matching")
print("=" * 60)

frame1 = get_first_frame(OUT1)
frame2 = get_first_frame(INPUT2)

ppm1 = pixel_per_metre(frame1, "video1 (corrected) – click 2 points of known distance")
ppm2 = pixel_per_metre(frame2, "video2 – click 2 points of known distance")

scale_factor = ppm1 / ppm2
H1, W1 = frame1.shape[:2]
H2, W2 = frame2.shape[:2]
new_w = int(round(W2 * scale_factor))
new_h = int(round(H2 * scale_factor))
print(f"\n  scale factor: {scale_factor:.4f}")
print(f"  video2 {W2}×{H2}  →  rescaled {new_w}×{new_h}  →  output {W1}×{H1}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Spatial anchor
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 3 – Spatial anchor alignment")
print("=" * 60)
print("Click ONE fixed landmark visible in both videos (e.g. a tank corner).")

anchor1 = pick_one_point(frame1, "video1 – click the anchor landmark")
print(f"  video1 anchor: {anchor1}")

frame2_preview = cv2.resize(frame2, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
anchor2 = pick_one_point(frame2_preview,
                         f"video2 (rescaled {new_w}×{new_h}) – click the SAME landmark")
print(f"  video2 anchor (rescaled space): {anchor2}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Temporal sync
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 4 – Temporal sync")
print("=" * 60)
print("Scrub each video to the frame where the object starts moving, press ENTER.")

sync1 = pick_sync_frame(OUT1,   "video1 (corrected) – motion start frame")
sync2 = pick_sync_frame(INPUT2, "video2 – motion start frame")
print(f"  video1 sync frame: {sync1}  |  video2 sync frame: {sync2}")

# Save for stack_videos.sh backward-compat (not used by this pipeline).
with open(ENVFILE, "w") as f:
    f.write(f"VIDEO1_START={sync1 / 30.0:.6f}\n")


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
total1s = int(cap1s.get(cv2.CAP_PROP_FRAME_COUNT))
writer1s = cv2.VideoWriter(OUT1_SYNCED, fourcc, 30, (W1, H1))
cap1s.set(cv2.CAP_PROP_POS_FRAMES, sync1)
n1 = total1s - sync1
print(f"Encoding {n1} frames (video1 from frame {sync1}) → {OUT1_SYNCED}")
for i in range(n1):
    ret, frame = cap1s.read()
    if not ret:
        break
    writer1s.write(frame)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{n1}", end="\r", flush=True)
cap1s.release()
writer1s.release()
print(f"\n  ✓ Saved {OUT1_SYNCED}")

# ── video2: rescaled + anchor-cropped + trimmed from sync2 ────────────────────
cap2 = cv2.VideoCapture(INPUT2)
total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
writer2 = cv2.VideoWriter(OUT2, fourcc, 30, (W1, H1))
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2)
n_frames = total2 - sync2
print(f"Encoding {n_frames} frames (video2 from frame {sync2}) → {OUT2}")

for i in range(n_frames):
    ret, frame = cap2.read()
    if not ret:
        break
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    final   = crop_with_anchor(resized, W1, H1, anchor2, anchor1)
    writer2.write(final)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{n_frames}", end="\r", flush=True)

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
    "-r", "30", "-i", OUT1_SYNCED,
    "-r", "30", "-i", OUT2,
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
