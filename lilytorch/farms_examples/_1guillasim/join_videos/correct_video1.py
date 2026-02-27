"""
Interactive perspective correction + horizontal flip for video1.

Usage:
    python correct_video1.py

Instructions:
    1. A window will open showing the first frame.
    2. Click exactly 4 points that define a region which should be a rectangle,
       going in order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT.
       (e.g. the four corners of the tank or any known rectangular feature)
    3. Press ENTER to confirm and start encoding.
    4. Press 'r' to reset your point selection.
    5. The corrected, horizontally-flipped video is saved as video1_corrected.mp4.
"""

import cv2
import numpy as np
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(DIR, "video1.MP4")
OUTPUT = os.path.join(DIR, "video1_corrected.mp4")

# ── step 1: grab first frame ──────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT)
if not cap.isOpened():
    sys.exit(f"Cannot open {INPUT}")

fps   = 30  # force 30 fps (GoPro 90k time-base workaround)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, first_frame = cap.read()
if not ret:
    sys.exit("Cannot read first frame")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ── step 2: interactive point selection ──────────────────────────────────────
points = []
LABELS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
COLORS = [(0,255,0), (0,255,255), (0,0,255), (255,0,0)]

def draw(img, pts):
    vis = img.copy()
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 8, COLORS[i], -1)
        cv2.putText(vis, LABELS[i], (p[0]+10, p[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[i], 2)
    if len(pts) == 4:
        cv2.polylines(vis, [np.array(pts)], True, (255,255,0), 2)
    next_label = LABELS[len(pts)] if len(pts) < 4 else "Press ENTER to confirm, 'r' to reset"
    cv2.putText(vis, f"Click: {next_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return vis

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.imshow("Select 4 corners", draw(first_frame, points))

cv2.namedWindow("Select 4 corners", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select 4 corners", min(W, 1280), min(H, 720))
cv2.setMouseCallback("Select 4 corners", mouse_cb)
cv2.imshow("Select 4 corners", draw(first_frame, points))

print("Click the 4 corners of a rectangle in the scene.")
print("Order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")
print("Press ENTER when done, 'r' to reset.")

while True:
    key = cv2.waitKey(50) & 0xFF
    if key == ord('r'):
        points.clear()
        cv2.imshow("Select 4 corners", draw(first_frame, points))
    elif key in (13, 10) and len(points) == 4:  # ENTER
        break
    elif key == 27:
        cap.release(); cv2.destroyAllWindows(); sys.exit("Cancelled")

cv2.destroyAllWindows()

# ── step 3: compute homography ────────────────────────────────────────────────
src = np.float32(points)

# destination: axis-aligned rectangle with the same width/height as the bounding box
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]
x0, x1 = min(x_coords), max(x_coords)
y0, y1 = min(y_coords), max(y_coords)
dst = np.float32([
    [x0, y0],   # top-left
    [x1, y0],   # top-right
    [x1, y1],   # bottom-right
    [x0, y1],   # bottom-left
])

H_mat, _ = cv2.findHomography(src, dst)
print(f"\nHomography matrix:\n{H_mat}\n")

# ── step 4: encode corrected + flipped video ──────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT, fourcc, fps, (W, H))

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {total} frames → {OUTPUT}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    warped  = cv2.warpPerspective(frame, H_mat, (W, H))
    flipped = cv2.flip(warped, 1)   # 1 = horizontal flip
    out.write(flipped)
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  {frame_idx}/{total}", end="\r", flush=True)

cap.release()
out.release()
print(f"\nDone! Saved: {OUTPUT}")
print("\nRe-run stack_videos.sh to rebuild the stacked video with the corrected top clip.")
