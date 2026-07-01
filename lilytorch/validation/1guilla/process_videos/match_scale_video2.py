"""
Two-point similarity alignment and temporal sync for video2.

Part of the join_videos pipeline.  Use this after crop_and_trim.py and before
the final ffmpeg stack step.

What it does
------------
  1. You click 2 corresponding landmarks (e.g. pool corners) in each video.
  2. A similarity transform (scale + rotation + translation) is computed so
     that the landmarks in video2 map exactly onto the landmarks in video1.
  3. A 50 % blended preview is shown so you can verify the alignment.
  4. Temporal sync – you scrub each video to the motion-start frame.
  5. The aligned + synced video2 is encoded at video1's frame size.

Usage
-----
    python match_scale_video2.py <video1_cropped> <video2_cropped>

Interactive steps
-----------------
  1. video1  – click TWO landmark points (e.g. two pool corners).
  2. video2  – click the SAME TWO landmarks.
  3. Inspect the blended preview; press ENTER to accept, ESC to cancel.
  4. video1  – scrub to motion start frame, press ENTER.
  5. video2  – scrub to motion start frame, press ENTER.

Output: video2_..._corrected.mp4, video_sync.env
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
def corrected_name(path):
    base, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(DIR, base + '_corrected' + ext)
OUTPUT = corrected_name(INPUT2)

# Read actual frame rates
_c1 = cv2.VideoCapture(INPUT1); FPS1 = _c1.get(cv2.CAP_PROP_FPS); _c1.release()
_c2 = cv2.VideoCapture(INPUT2); FPS2 = _c2.get(cv2.CAP_PROP_FPS); _c2.release()
if FPS1 <= 0: FPS1 = 30.0
if FPS2 <= 0: FPS2 = 30.0

print(f"Reference (video1): {INPUT1}  ({FPS1:.2f} fps)")
print(f"To scale  (video2): {INPUT2}  ({FPS2:.2f} fps)")


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


# ── pick 2 corresponding landmarks in each video ─────────────────────────────

frame1 = get_first_frame(INPUT1)
frame2 = get_first_frame(INPUT2)
H1, W1 = frame1.shape[:2]

print("\n=== Landmark selection ===")
print("Click the SAME 2 fixed landmarks (e.g. two pool corners) in both videos.")

pts_v1 = pick_two_points(frame1, "video1 – click 2 landmarks")
pts_v2 = pick_two_points(frame2, "video2 – click the SAME 2 landmarks")

print(f"  video1 landmarks: {pts_v1}")
print(f"  video2 landmarks: {pts_v2}")

# ── compute similarity transform (video2 → video1) ───────────────────────────

src = np.float32(pts_v2)   # points in video2
dst = np.float32(pts_v1)   # where they should land (video1 coords)

M, _ = cv2.estimateAffinePartial2D(src, dst)
if M is None:
    sys.exit("Could not compute transform from the given points.")

s = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
print(f"\n  Scale factor : {s:.4f}")
print(f"  Rotation     : {theta:.2f}°")
print(f"  Translation  : ({M[0, 2]:.1f}, {M[1, 2]:.1f}) px")

# ── blended preview ───────────────────────────────────────────────────────────

preview = cv2.warpAffine(frame2, M, (W1, H1))
blend = cv2.addWeighted(frame1, 0.5, preview, 0.5, 0)

title = "Preview (50% blend) – ENTER to accept, ESC to cancel"
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title, min(W1, 1280), min(H1, 720))
cv2.imshow(title, blend)

while True:
    key = cv2.waitKey(50) & 0xFF
    if key in (13, 10):
        break
    elif key == 27:
        cv2.destroyAllWindows()
        sys.exit("Cancelled – re-run and pick better landmarks.")
cv2.destroyAllWindows()

# ── temporal sync ─────────────────────────────────────────────────────────────
print("\n=== Temporal sync ===")
print("Scrub each video to the frame where the object starts moving, then press ENTER.")
sync1 = pick_sync_frame(INPUT1, "video1 – scrub to motion start, press ENTER")
sync2 = pick_sync_frame(INPUT2, "video2 – scrub to motion start, press ENTER")
print(f"  video1 sync frame: {sync1},  video2 sync frame: {sync2}")

# Write video1 sync offset using actual video1 FPS
video1_start_sec = sync1 / FPS1
env_path = os.path.join(DIR, "video_sync.env")
with open(env_path, "w") as f:
    f.write(f"VIDEO1_START={video1_start_sec:.6f}\n")
print(f"  Saved video1 start offset → {env_path}")

# ── encode aligned video2 ────────────────────────────────────────────────────
cap2 = cv2.VideoCapture(INPUT2)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT, fourcc, FPS2, (W1, H1))

total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
cap2.set(cv2.CAP_PROP_POS_FRAMES, sync2)
n_frames = total - sync2
print(f"\nProcessing {n_frames} frames (video2 from frame {sync2}) …")

idx = 0
while True:
    ret, frame = cap2.read()
    if not ret:
        break
    final = cv2.warpAffine(frame, M, (W1, H1))
    out.write(final)
    idx += 1
    if idx % 50 == 0:
        print(f"  {idx}/{n_frames}", end="\r", flush=True)

cap2.release()
out.release()
print(f"\nDone! Saved: {OUTPUT}")
