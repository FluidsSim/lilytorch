"""
Correct an experimental video: trim to start/end frames, optionally apply
perspective correction, and optionally flip horizontally.

Usage
-----
    python correct_video1.py <video> [--flip] [--no-homography]

Steps
-----
  1. Scrub to the motion-start frame and motion-end frame.
  2. (unless --no-homography) Pick 4 corners of a known rectangle →
     homography warp.  The full image canvas is preserved (no cropping).
  3. Encode the corrected video trimmed to [start, end].

Output
------
    <video>_corrected.<ext>   in the same directory as the input.
"""

import argparse
import cv2
import numpy as np
import sys
import os
from video_utils import pick_sync_frame, pick_four_points, compute_homography


# ── parse arguments ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Perspective-correct and trim an experimental video.")
parser.add_argument("video", help="Experimental recording (e.g. GoPro .MP4)")
parser.add_argument("--flip", action="store_true",
                    help="Horizontally flip after perspective correction")
parser.add_argument("--no-homography", action="store_true",
                    help="Skip the perspective-correction step (trim + flip only)")
args = parser.parse_args()

INPUT = os.path.abspath(args.video)
FLIP  = args.flip
SKIP_HOMOGRAPHY = args.no_homography
DIR   = os.path.dirname(INPUT)

base, ext = os.path.splitext(os.path.basename(INPUT))
OUTPUT = os.path.join(DIR, base + "_corrected" + ext)

if not os.path.exists(INPUT):
    sys.exit(f"File not found: {INPUT}")

cap_info = cv2.VideoCapture(INPUT)
FPS = cap_info.get(cv2.CAP_PROP_FPS)
cap_info.release()

print(f"Input : {INPUT}")
print(f"FPS   : {FPS:.2f}")
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


def pick_four_points(frame, title):
    """Click 4 corners of a rectangle, then drag to adjust.  Scroll to zoom.

    Returns list of 4 (x, y) in **original-image coordinates**.

    Controls
    --------
    - Click to place each corner (TL → TR → BR → BL).
    - After all 4 are placed, left-click-drag any point to reposition.
    - Scroll wheel : zoom in / out (centred on cursor).
    - Right-click-drag : pan the view.
    - 'r'     : reset all points.
    - 'z'     : reset zoom / pan (fit to window).
    - ENTER   : confirm current positions.
    - ESC     : cancel.
    """
    LABELS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 0)]
    GRAB_RADIUS_IMG = 18      # grab radius in *image* pixels (scales w/ zoom)
    pts = []                  # points in original-image coords
    drag = {"idx": -1}

    # ── view state (zoom + pan) ──────────────────────────────────────────
    h_img, w_img = frame.shape[:2]
    # Display size (window size)
    disp_w, disp_h = min(w_img, 1280), min(h_img, 720)
    # View: which rectangle of the original image is visible.
    # Stored as floats for sub-pixel panning.
    view = {"x": 0.0, "y": 0.0, "w": float(w_img), "h": float(h_img)}
    pan  = {"active": False, "sx": 0, "sy": 0, "vx0": 0.0, "vy0": 0.0}

    def clamp_view():
        """Keep the view rectangle inside the image bounds."""
        view["x"] = max(0.0, min(view["x"], w_img - view["w"]))
        view["y"] = max(0.0, min(view["y"], h_img - view["h"]))

    def img2screen(px, py):
        """Original-image coords → display coords."""
        sx = int((px - view["x"]) / view["w"] * disp_w)
        sy = int((py - view["y"]) / view["h"] * disp_h)
        return sx, sy

    def screen2img(sx, sy):
        """Display coords → original-image coords."""
        ix = view["x"] + sx / disp_w * view["w"]
        iy = view["y"] + sy / disp_h * view["h"]
        return ix, iy

    def scale_factor():
        """Current pixels-per-image-pixel."""
        return disp_w / view["w"]

    # ── drawing ──────────────────────────────────────────────────────────
    def draw(pts, highlight_idx=-1):
        # Crop & scale the visible region
        x0 = int(max(view["x"], 0))
        y0 = int(max(view["y"], 0))
        x1 = int(min(view["x"] + view["w"], w_img))
        y1 = int(min(view["y"] + view["h"], h_img))
        crop = frame[y0:y1, x0:x1]
        vis = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        sf = scale_factor()
        # draw connecting polygon
        if len(pts) == 4:
            poly = np.array([img2screen(*p) for p in pts], dtype=np.int32)
            cv2.polylines(vis, [poly], True, (255, 255, 0), 2)
        # draw each corner
        for i, p in enumerate(pts):
            sp = img2screen(*p)
            r = max(4, int((12 if i == highlight_idx else 8) * min(sf, 3)))
            cv2.circle(vis, sp, r, COLORS[i], -1)
            cv2.putText(vis, LABELS[i], (sp[0]+10, sp[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        min(0.8 * sf, 2.0), COLORS[i], max(1, int(2*min(sf,2))))
        # instruction bar (always same screen size)
        if len(pts) < 4:
            msg = f"Click: {LABELS[len(pts)]}  |  scroll=zoom  Rdrag=pan"
        else:
            msg = "Drag pts | scroll=zoom | Rdrag=pan | ENTER | r=reset | z=unzoom"
        cv2.putText(vis, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # zoom indicator
        zoom_pct = int(w_img / view["w"] * 100)
        cv2.putText(vis, f"Zoom {zoom_pct}%", (10, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return vis

    def refresh():
        cv2.imshow(title, draw(pts, drag["idx"]))

    # ── nearest point (in image coords) ──────────────────────────────────
    def nearest_point(ix, iy):
        best, best_d = -1, GRAB_RADIUS_IMG / scale_factor() + GRAB_RADIUS_IMG
        for i, p in enumerate(pts):
            d = np.hypot(ix - p[0], iy - p[1])
            if d < best_d:
                best, best_d = i, d
        return best

    # ── mouse callback ───────────────────────────────────────────────────
    def mouse_cb(event, sx, sy, flags, param):
        ix, iy = screen2img(sx, sy)

        # ── scroll to zoom ───────────────────────────────────────────────
        if event == cv2.EVENT_MOUSEWHEEL:
            zfactor = 1.15 if flags > 0 else 1 / 1.15
            new_w = max(disp_w * 0.5, min(w_img, view["w"] / zfactor))
            new_h = max(disp_h * 0.5, min(h_img, view["h"] / zfactor))
            # keep the image point under the cursor stationary
            frac_x = sx / disp_w
            frac_y = sy / disp_h
            view["x"] = ix - frac_x * new_w
            view["y"] = iy - frac_y * new_h
            view["w"] = new_w
            view["h"] = new_h
            clamp_view()
            refresh()
            return

        # ── right-click drag → pan ───────────────────────────────────────
        if event == cv2.EVENT_RBUTTONDOWN:
            pan["active"] = True
            pan["sx"], pan["sy"] = sx, sy
            pan["vx0"], pan["vy0"] = view["x"], view["y"]
            return
        if event == cv2.EVENT_MOUSEMOVE and pan["active"]:
            dx = (sx - pan["sx"]) / disp_w * view["w"]
            dy = (sy - pan["sy"]) / disp_h * view["h"]
            view["x"] = pan["vx0"] - dx
            view["y"] = pan["vy0"] - dy
            clamp_view()
            refresh()
            return
        if event == cv2.EVENT_RBUTTONUP:
            pan["active"] = False
            return

        # ── left-click: place / drag points ──────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 4:
                pts.append((int(round(ix)), int(round(iy))))
            else:
                drag["idx"] = nearest_point(ix, iy)
            refresh()

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if drag["idx"] >= 0:
                pts[drag["idx"]] = (int(round(ix)), int(round(iy)))
                refresh()

        elif event == cv2.EVENT_LBUTTONUP:
            if drag["idx"] >= 0:
                pts[drag["idx"]] = (int(round(ix)), int(round(iy)))
                drag["idx"] = -1
                refresh()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp_w, disp_h)
    cv2.setMouseCallback(title, mouse_cb)
    refresh()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('r'):
            pts.clear(); drag["idx"] = -1; refresh()
        elif key == ord('z'):
            # reset zoom
            view["x"], view["y"] = 0.0, 0.0
            view["w"], view["h"] = float(w_img), float(h_img)
            refresh()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows(); sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Pick start and end frames
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Pick start and end frames")
print("=" * 60)

print("Scrub to the frame where the object STARTS moving, press ENTER.")
start_frame = pick_sync_frame(INPUT, "video — motion START frame")
print(f"  start frame: {start_frame}")

print("\nScrub to the frame where you want the video to END, press ENTER.")
end_frame = pick_sync_frame(INPUT, "video — END frame")
print(f"  end frame:   {end_frame}")

if end_frame <= start_frame:
    sys.exit(f"End frame ({end_frame}) must be after start frame ({start_frame})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Perspective correction (+ optional flip)
# ─────────────────────────────────────────────────────────────────────────────

H_full = None   # will stay None when homography is skipped

if SKIP_HOMOGRAPHY:
    print()
    print("=" * 60)
    print("STEP 2 — Perspective correction SKIPPED (--no-homography)")
    print("=" * 60)
    # Read one frame just to get the resolution
    cap_pick = cv2.VideoCapture(INPUT)
    cap_pick.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret_pick, frame_raw = cap_pick.read()
    H_px, W_px = frame_raw.shape[:2]
    cap_pick.release()
    if not ret_pick:
        sys.exit(f"Cannot read frame {start_frame} from {INPUT}")
    out_w, out_h = W_px, H_px
else:
    print()
    print("=" * 60)
    print(f"STEP 2 — Perspective correction{' + flip' if FLIP else ''}")
    print("=" * 60)

    cap_pick = cv2.VideoCapture(INPUT)
    cap_pick.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret_pick, frame_raw = cap_pick.read()
    H_px, W_px = frame_raw.shape[:2]
    cap_pick.release()
    if not ret_pick:
        sys.exit(f"Cannot read frame {start_frame} from {INPUT}")

    print(f"Showing frame {start_frame} for corner picking.")
    print("Click the 4 corners of a rectangular feature.")
    print("Order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")
    corners = pick_four_points(frame_raw, "video — click 4 rectangle corners")

    src_pts = np.float32(corners)
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dst_pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    H_mat, _ = cv2.findHomography(src_pts, dst_pts)

    # Full canvas — warp the 4 image corners to find the bounding box so
    # nothing is cropped off.
    img_corners = np.float32(
        [[0, 0], [W_px, 0], [W_px, H_px], [0, H_px]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(img_corners, H_mat).reshape(-1, 2)
    all_x = warped_corners[:, 0]; all_y = warped_corners[:, 1]
    min_x = int(np.floor(all_x.min())); max_x = int(np.ceil(all_x.max()))
    min_y = int(np.floor(all_y.min())); max_y = int(np.ceil(all_y.max()))
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float64)
    H_full = T @ H_mat
    out_w = max_x - min_x
    out_h = max_y - min_y
    print(f"  Perspective warp: {W_px}×{H_px} → {out_w}×{out_h}  (full canvas)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Encode corrected + trimmed video
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 3 — Encoding")
print("=" * 60)

n_frames = end_frame - start_frame
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (out_w, out_h))

cap = cv2.VideoCapture(INPUT)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

print(f"Encoding {n_frames} frames (frames {start_frame}–{end_frame}) → {OUTPUT}")
for i in range(n_frames):
    ret, frame = cap.read()
    if not ret:
        break
    if H_full is not None:
        frame = cv2.warpPerspective(frame, H_full, (out_w, out_h))
    out_frame = cv2.flip(frame, 1) if FLIP else frame
    writer.write(out_frame)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_frames}", end="\r", flush=True)

cap.release()
writer.release()

duration = n_frames / FPS
print(f"\n  ✓ Saved {OUTPUT}")
print(f"  {n_frames} frames at {FPS:.1f} fps = {duration:.2f} s")
print()
print("Next: run  python join_videos.py <corrected_video1> <video2>")
print("=" * 60)
