"""
Metric scale matching, spatial anchoring, and temporal sync for video2.

Part of the join_videos pipeline (standalone version of pipeline.py steps 2-4
plus video2 encoding).  Use this after correct_video1.py and before
stack_videos.sh.

What it does
------------
  1. Metric scale – you click 2 points of known distance in each video and
     type the real-world distance in metres.  video2 is rescaled so that
     1 m spans the same number of pixels as in video1.
  2. Spatial anchor – you click one fixed landmark visible in both videos
     (e.g. a tank corner).  video2 is cropped/padded so the landmark sits
     at the same pixel position as in video1.
  3. Temporal sync – you scrub each video to the frame where motion begins.
     video2 is trimmed from that frame onward; the video1 start offset is
     written to video_sync.env for stack_videos.sh.
  4. Encodes the result as video2_scaled.mp4 matching video1's frame size.

Usage
-----
    python match_scale_video2.py

Interactive steps
-----------------
  1. video1  – click TWO points along a known distance, type distance (m).
  2. video2  – click TWO points along the same kind of known distance, type distance (m).
  3. video1  – click ONE landmark point (e.g. a fixed corner of the tank).
  4. video2  – click the SAME landmark point in video2 (after rescaling preview).
  5. video1  – scrub to motion start frame, press ENTER.
  6. video2  – scrub to motion start frame, press ENTER.

Input:  video1_corrected.mp4 (or video1.MP4), video2.mp4
Output: video2_scaled.mp4, video_sync.env

After this, run stack_videos.sh to produce the final stacked comparison.
"""

import cv2
import numpy as np
import sys
import os


import argparse

parser = argparse.ArgumentParser(description="Scale, anchor, and sync two videos.")
parser.add_argument("input1", help="Corrected experiment video filename (e.g. exp_water_corrected.MP4)")
parser.add_argument("input2", help="Corrected simulation video filename (e.g. sim_water_corrected.mp4)")
args = parser.parse_args()

INPUT1 = os.path.abspath(args.input1)
INPUT2 = os.path.abspath(args.input2)
DIR = os.path.dirname(INPUT1)
def synced_name(path):
    base, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(DIR, base + '_synced' + ext)
OUTPUT = INPUT2  # Overwrite/corrected video2

print(f"Reference (video1): {INPUT1}")
print(f"To scale  (video2): {INPUT2}")


# ── helpers ───────────────────────────────────────────────────────────────────

def get_first_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Cannot read first frame from {path}")
    return frame


def pick_two_points(frame, title):
    """Open window, let user click 2 points. Returns list of 2 (x,y) tuples."""
    pts = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            vis = draw(frame, pts)
            cv2.imshow(title, vis)

    def draw(img, pts):
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 8, (0, 255, 0), -1)
            cv2.putText(vis, f"P{i+1}", (p[0]+10, p[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], (0, 255, 255), 2)
            px_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            cv2.putText(vis, f"{px_dist:.1f} px", (
                (pts[0][0]+pts[1][0])//2 + 10,
                (pts[0][1]+pts[1][1])//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        next_action = f"Click P{len(pts)+1}" if len(pts) < 2 else "Press ENTER to confirm, 'r' to reset"
        cv2.putText(vis, next_action, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return vis

    h, w = frame.shape[:2]
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(title, mouse_cb)
    cv2.imshow(title, draw(frame, pts))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            pts.clear()
            cv2.imshow(title, draw(frame, pts))
        elif key in (13, 10) and len(pts) == 2:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts


def pixel_per_metre(frame, title):
    print(f"\n=== {title} ===")
    print("Click 2 points along a known real-world distance.")
    pts = pick_two_points(frame, title)
    px_dist = float(np.linalg.norm(np.array(pts[0]) - np.array(pts[1])))
    while True:
        try:
            m_dist = float(input(f"  Distance between the two points in metres: "))
            if m_dist > 0:
                break
        except ValueError:
            pass
    ppm = px_dist / m_dist
    print(f"  → {px_dist:.1f} px = {m_dist} m  →  {ppm:.2f} px/m")
    return ppm


def pick_one_point(frame, title):
    """Open window, let user click exactly 1 point (re-clickable). Returns (x, y)."""
    pts = []

    def draw(img, pts):
        vis = img.copy()
        if pts:
            cv2.circle(vis, pts[0], 10, (0, 0, 255), -1)
            cv2.putText(vis, "Anchor", (pts[0][0]+10, pts[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        action = "Click the anchor point" if not pts else "Press ENTER to confirm, 'r' to reset"
        cv2.putText(vis, action, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return vis

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.clear()
            pts.append((x, y))
            cv2.imshow(title, draw(frame, pts))

    h, w = frame.shape[:2]
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(title, mouse_cb)
    cv2.imshow(title, draw(frame, pts))

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            pts.clear()
            cv2.imshow(title, draw(frame, pts))
        elif key in (13, 10) and pts:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts[0]


def pick_sync_frame(path, title):
    """Show video with a trackbar scrubber. ENTER selects the current frame index."""
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
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(title, vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):   # ENTER
            cap.release()
            cv2.destroyAllWindows()
            return idx
        elif key == ord('r'):
            state["idx"] = 0
            cv2.setTrackbarPos("Frame", title, 0)
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Cancelled")


def crop_with_anchor(frame, target_w, target_h, anchor_src, anchor_dst):
    """
    Crop/pad `frame` (already rescaled) so that pixel `anchor_src` in the
    rescaled frame lands exactly on pixel `anchor_dst` in the output frame.

    anchor_src : (x, y) – landmark position in the rescaled video2 frame
    anchor_dst : (x, y) – where that landmark must appear in the output
                          (= its position in video1)
    """
    h, w = frame.shape[:2]
    offset_x = anchor_src[0] - anchor_dst[0]
    offset_y = anchor_src[1] - anchor_dst[1]

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    src_x0 = max(0, offset_x);  src_y0 = max(0, offset_y)
    src_x1 = min(w, offset_x + target_w)
    src_y1 = min(h, offset_y + target_h)
    dst_x0 = max(0, -offset_x); dst_y0 = max(0, -offset_y)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return canvas


# ── measure both scales ───────────────────────────────────────────────────────

frame1 = get_first_frame(INPUT1)
frame2 = get_first_frame(INPUT2)

ppm1 = pixel_per_metre(frame1, "video1 – click 2 points of known distance")
ppm2 = pixel_per_metre(frame2, "video2 – click 2 points of known distance")

scale_factor = ppm1 / ppm2
print(f"\nScale factor to apply to video2: {scale_factor:.4f}  "
      f"(video2 will be {'enlarged' if scale_factor>1 else 'shrunk'} by {scale_factor:.2f}x)")

# target size = video1 frame size
H1, W1 = frame1.shape[:2]
H2, W2 = frame2.shape[:2]

new_w = int(round(W2 * scale_factor))
new_h = int(round(H2 * scale_factor))
print(f"video2 original: {W2}×{H2}  →  rescaled: {new_w}×{new_h}  →  final: {W1}×{H1}")

# ── pick anchor points ────────────────────────────────────────────────────────
print("\n=== Anchor alignment ===")
print("Click ONE fixed landmark visible in both videos (e.g. a tank corner).")

anchor_in_video1 = pick_one_point(
    frame1, "video1 – click the anchor landmark")
print(f"  video1 anchor: {anchor_in_video1}")

frame2_rescaled_preview = cv2.resize(frame2, (new_w, new_h),
                                     interpolation=cv2.INTER_LANCZOS4)
anchor_in_video2_rescaled = pick_one_point(
    frame2_rescaled_preview,
    f"video2 (rescaled {new_w}×{new_h}) – click the SAME landmark")
print(f"  video2 anchor (rescaled space): {anchor_in_video2_rescaled}")

# ── temporal sync ────────────────────────────────────────────────────────────
print("\n=== Temporal sync ===")
print("Scrub each video to the frame where the object starts moving, then press ENTER.")
sync1 = pick_sync_frame(INPUT1, "video1 – scrub to motion start, press ENTER")
sync2 = pick_sync_frame(INPUT2, "video2 – scrub to motion start, press ENTER")
print(f"  video1 sync frame: {sync1},  video2 sync frame: {sync2}")

# Write video1 sync offset so stack_videos.sh can apply -ss
FPS = 30.0
video1_start_sec = sync1 / FPS
env_path = os.path.join(DIR, "video_sync.env")
with open(env_path, "w") as f:
    f.write(f"VIDEO1_START={video1_start_sec:.6f}\n")
print(f"  Saved video1 start offset → {env_path}")

# ── encode scaled video2 ──────────────────────────────────────────────────────
cap2 = cv2.VideoCapture(INPUT2)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT, fourcc, 30, (W1, H1))

total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

# Seek both videos to their respective sync frames.
# ffmpeg trims video1 with -ss, so video2_scaled must also start at sync2.
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2)
n_frames = total - sync2
print(f"\nProcessing {n_frames} frames (video2 from frame {sync2}) …")

idx = 0
while True:
    ret, frame = cap2.read()
    if not ret:
        break
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    final   = crop_with_anchor(resized, W1, H1,
                               anchor_in_video2_rescaled, anchor_in_video1)
    out.write(final)
    idx += 1
    if idx % 50 == 0:
        print(f"  {idx}/{total}", end="\r", flush=True)

cap2.release()
out.release()
print(f"\nDone! Saved: {OUTPUT}")
print("Now run:  bash stack_videos.sh")
