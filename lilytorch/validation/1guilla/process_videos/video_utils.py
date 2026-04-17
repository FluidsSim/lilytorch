"""
Shared GUI helpers for video processing scripts.

Provides:
  - pick_sync_frame  – scrubber window to select a frame index
  - pick_four_points – click/drag 4 corners with zoom + pan support
"""

import sys
import cv2
import numpy as np


def pick_sync_frame(path: str, title: str) -> int:
    """Scrubber window.  ENTER selects the current frame index."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first = cap.read()
    if not ret:
        sys.exit(f"Cannot read first frame from {path}")
    h0, w0 = first.shape[:2]

    state = {"idx": 0, "frame": first.copy()}

    def on_trackbar(val):
        state["idx"] = val

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


def pick_four_points(frame: np.ndarray, title: str) -> list[tuple[int, int]]:
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
    GRAB_RADIUS_IMG = 18
    pts = []
    drag = {"idx": -1}

    h_img, w_img = frame.shape[:2]
    disp_w, disp_h = min(w_img, 1280), min(h_img, 720)
    view = {"x": 0.0, "y": 0.0, "w": float(w_img), "h": float(h_img)}
    pan  = {"active": False, "sx": 0, "sy": 0, "vx0": 0.0, "vy0": 0.0}

    def clamp_view():
        view["x"] = max(0.0, min(view["x"], w_img - view["w"]))
        view["y"] = max(0.0, min(view["y"], h_img - view["h"]))

    def img2screen(px, py):
        sx = int((px - view["x"]) / view["w"] * disp_w)
        sy = int((py - view["y"]) / view["h"] * disp_h)
        return sx, sy

    def screen2img(sx, sy):
        ix = view["x"] + sx / disp_w * view["w"]
        iy = view["y"] + sy / disp_h * view["h"]
        return ix, iy

    def scale_factor():
        return disp_w / view["w"]

    def draw(pts, highlight_idx=-1):
        x0 = int(max(view["x"], 0))
        y0 = int(max(view["y"], 0))
        x1 = int(min(view["x"] + view["w"], w_img))
        y1 = int(min(view["y"] + view["h"], h_img))
        crop = frame[y0:y1, x0:x1]
        vis = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        sf = scale_factor()
        if len(pts) == 4:
            poly = np.array([img2screen(*p) for p in pts], dtype=np.int32)
            cv2.polylines(vis, [poly], True, (255, 255, 0), 2)
        for i, p in enumerate(pts):
            sp = img2screen(*p)
            r = max(4, int((12 if i == highlight_idx else 8) * min(sf, 3)))
            cv2.circle(vis, sp, r, COLORS[i], -1)
            cv2.putText(vis, LABELS[i], (sp[0]+10, sp[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        min(0.8 * sf, 2.0), COLORS[i], max(1, int(2*min(sf,2))))
        if len(pts) < 4:
            msg = f"Click: {LABELS[len(pts)]}  |  scroll=zoom  Rdrag=pan"
        else:
            msg = "Drag pts | scroll=zoom | Rdrag=pan | ENTER | r=reset | z=unzoom"
        cv2.putText(vis, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        zoom_pct = int(w_img / view["w"] * 100)
        cv2.putText(vis, f"Zoom {zoom_pct}%", (10, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return vis

    def refresh():
        cv2.imshow(title, draw(pts, drag["idx"]))

    def nearest_point(ix, iy):
        best, best_d = -1, GRAB_RADIUS_IMG / scale_factor() + GRAB_RADIUS_IMG
        for i, p in enumerate(pts):
            d = np.hypot(ix - p[0], iy - p[1])
            if d < best_d:
                best, best_d = i, d
        return best

    def mouse_cb(event, sx, sy, flags, param):
        ix, iy = screen2img(sx, sy)

        if event == cv2.EVENT_MOUSEWHEEL:
            zfactor = 1.15 if flags > 0 else 1 / 1.15
            new_w = max(disp_w * 0.5, min(w_img, view["w"] / zfactor))
            new_h = max(disp_h * 0.5, min(h_img, view["h"] / zfactor))
            frac_x = sx / disp_w
            frac_y = sy / disp_h
            view["x"] = ix - frac_x * new_w
            view["y"] = iy - frac_y * new_h
            view["w"] = new_w
            view["h"] = new_h
            clamp_view()
            refresh()
            return

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
            pts.clear()
            drag["idx"] = -1
            refresh()
        elif key == ord('z'):
            view["x"], view["y"] = 0.0, 0.0
            view["w"], view["h"] = float(w_img), float(h_img)
            refresh()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit("Cancelled")

    cv2.destroyAllWindows()
    return pts


def compute_homography(frame: np.ndarray,
                       corners: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    """
    Given 4 corners (TL, TR, BR, BL) clicked by the user, compute a
    full-canvas perspective-correction homography.

    Returns (H_full, out_w, out_h).
    """
    H_px, W_px = frame.shape[:2]
    src_pts = np.float32(corners)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dst_pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    H_mat, _ = cv2.findHomography(src_pts, dst_pts)

    img_corners = np.float32(
        [[0, 0], [W_px, 0], [W_px, H_px], [0, H_px]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(img_corners, H_mat).reshape(-1, 2)
    min_x = int(np.floor(warped[:, 0].min()))
    max_x = int(np.ceil(warped[:, 0].max()))
    min_y = int(np.floor(warped[:, 1].min()))
    max_y = int(np.ceil(warped[:, 1].max()))
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float64)
    H_full = T @ H_mat
    return H_full, max_x - min_x, max_y - min_y
