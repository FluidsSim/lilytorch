"""
Interactive crop-region and end-time picker for two videos.

Usage
-----
    python crop_and_trim.py <video1> <video2>

Steps:
  1. Draw crop rectangle on video1 (applied to both).
  2. Scrub video1 to choose end time (applied to both).
  3. Encode cropped + trimmed versions of both videos.

Output
------
  <video1>_cropped.<ext>  – cropped + trimmed video1
  <video2>_cropped.<ext>  – cropped + trimmed video2
"""

import argparse
import cv2
import os
import sys

parser = argparse.ArgumentParser(description="Interactive crop & trim picker.")
parser.add_argument("video1", help="First video (experiment)")
parser.add_argument("video2", help="Second video (simulation)")
args = parser.parse_args()

V1 = os.path.abspath(args.video1)
V2 = os.path.abspath(args.video2)
DIR = os.path.dirname(V1)

def cropped_name(path):
    base, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(os.path.abspath(path)), base + '_cropped' + ext)


def get_first_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Cannot read first frame from {path}")
    return frame


def pick_crop(frame, title):
    """Let user draw a crop rectangle. Returns (x, y, w, h)."""
    print(f"\n=== {title} ===")
    print("Draw a rectangle for the crop region, then press ENTER/SPACE to confirm.")
    print("Press 'c' to cancel (no crop).")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    h0, w0 = frame.shape[:2]
    cv2.resizeWindow(title, min(w0, 1280), min(h0, 720))
    roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("  No crop selected – using full frame.")
        return 0, 0, frame.shape[1], frame.shape[0]
    print(f"  Crop: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


def pick_end_frame(path, title):
    """Scrubber to pick the last frame. Returns time in seconds."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    state = {"idx": total - 1}

    print(f"\n=== {title} ===")
    print("Scrub to the LAST frame you want to keep, then press ENTER.")

    def on_trackbar(val):
        state["idx"] = val

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    h0, w0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv2.resizeWindow(title, min(w0, 1280), min(h0, 720))
    cv2.createTrackbar("Frame", title, total - 1, total - 1, on_trackbar)

    last_idx = -1
    while True:
        idx = state["idx"]
        if idx != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                t = idx / fps
                vis = frame.copy()
                cv2.putText(vis, f"Frame {idx}/{total-1}  t={t:.2f}s  [ENTER=confirm]",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(title, vis)
            last_idx = idx

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            break
        elif key == 27:
            cv2.destroyAllWindows()
            cap.release()
            sys.exit("Cancelled")

    cv2.destroyAllWindows()
    cap.release()
    end_time = state["idx"] / fps
    print(f"  End frame: {state['idx']}  →  {end_time:.3f}s")
    return end_time


# ── Video 1 – pick crop and end time ─────────────────────────────────────────
frame1 = get_first_frame(V1)
h1, w1 = frame1.shape[:2]
cx, cy, cw, ch = pick_crop(frame1, "Crop video1 (applied to both)")
end_time = pick_end_frame(V1, "End-time video1 (applied to both)")

# Scale crop rectangle to video2's resolution
frame2 = get_first_frame(V2)
h2, w2 = frame2.shape[:2]
sx, sy = w2 / w1, h2 / h1
c2x = int(cx * sx)
c2y = int(cy * sy)
c2w = int(cw * sx)
c2h = int(ch * sy)
# Ensure even dimensions and stay within bounds
c2w = min(c2w - c2w % 2, w2 - c2x)
c2h = min(c2h - c2h % 2, h2 - c2y)

print(f"\nVideo1 crop: x={cx}, y={cy}, w={cw}, h={ch}  ({w1}x{h1})")
print(f"Video2 crop: x={c2x}, y={c2y}, w={c2w}, h={c2h}  ({w2}x{h2})")
print(f"End time: {end_time:.3f}s")


import subprocess


def encode_cropped(src, dst, cx, cy, cw, ch, end_time):
    """Use ffmpeg to crop and trim src → dst."""
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-t", f"{end_time:.3f}",
        "-filter:v", f"crop={cw}:{ch}:{cx}:{cy}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        dst,
    ]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  → {dst}")


OUT1 = cropped_name(V1)
OUT2 = cropped_name(V2)

print(f"\nEncoding cropped video1 …")
encode_cropped(V1, OUT1, cx, cy, cw, ch, end_time)

print(f"Encoding cropped video2 …")
encode_cropped(V2, OUT2, c2x, c2y, c2w, c2h, end_time)

print(f"\nDone!")
