#!/usr/bin/env python3
"""
Generate MP4 videos from PNG frames saved by the NS solver.

Usage
-----
    python video_from_png.py <run_dir>              # one specific run
    python video_from_png.py <save_path>            # auto-pick latest run
    python video_from_png.py <run_dir> --fields omega_z_3d pressure_3d
    python video_from_png.py <run_dir> --fps 30     # override frame-rate

The script reads ``parameters.yaml`` from the run directory to extract
``dt`` and ``save_every``, then discovers every sub-folder containing
PNGs and generates one H.264 MP4 per field.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


# ── helpers ──────────────────────────────────────────────────────────

def _latest_run(save_path: str) -> Path:
    """Return the most-recent timestamped sub-directory under *save_path*."""
    p = Path(save_path)
    runs = sorted(
        [d for d in p.iterdir() if d.is_dir() and re.match(r"\d{4}-", d.name)],
        key=lambda d: d.name,
    )
    if not runs:
        sys.exit(f"No timestamped run directories found under {p}")
    return runs[-1]


def _discover_fields(run_dir: Path):
    """Yield (field_name, sorted list of PNG paths) for every sub-folder."""
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        pngs = sorted(sub.glob("*.png"))
        if pngs:
            yield sub.name, pngs


def _iteration_from_name(path: Path) -> int:
    """Extract the iteration number from e.g. ``omega_z_3d_000100.png``."""
    m = re.search(r"_(\d+)\.png$", path.name)
    return int(m.group(1)) if m else 0


def _read_parameters(run_dir: Path):
    """Read parameters.yaml and return (dt, save_every)."""
    params_file = run_dir / "parameters.yaml"
    if not params_file.exists():
        return None, None
    with open(params_file) as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError:
            # parameters.yaml may contain serialised Python objects
            # (e.g. FARMS options) that safe_load cannot handle.
            # Fall back to unsafe_load which supports arbitrary tags.
            f.seek(0)
            try:
                params = yaml.unsafe_load(f)
            except Exception:
                return None, None
    dt = params.get("solver", {}).get("dt", None)
    save_every = params.get("output", {}).get("save_every", None)
    return dt, save_every


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ── main ─────────────────────────────────────────────────────────────

def make_videos(
    run_dir: Path,
    fields: list[str] | None = None,
    fps: float | None = None,
    slow_factor: float = 1.0,
    time_overlay: bool = True,
    crf: int = 18,
):
    """
    Generate one MP4 video per field sub-directory.

    Parameters
    ----------
    run_dir : Path
        The timestamped run directory containing sub-folders of PNGs.
    fields : list[str] or None
        If given, only generate videos for these sub-folders.
    fps : float or None
        Override the frame-rate.  When *None*, computed from
        ``slow_factor / (save_every * dt)`` using parameters.yaml.
    slow_factor : float
        Real-time multiplier.  1.0 = physical time equals video time.
    time_overlay : bool
        Burn a timestamp into each frame (requires ffmpeg drawtext).
    crf : int
        H.264 quality (lower = better, 18 ≈ visually lossless).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    # ── simulation parameters ──
    dt, save_every = _read_parameters(run_dir)
    if dt is not None and save_every is not None:
        print(f"parameters.yaml → dt={dt}, save_every={save_every}")
        if fps is None:
            fps = slow_factor / (save_every * dt)
    if fps is None:
        fps = 10.0
        print(f"  (could not determine FPS; defaulting to {fps})")
    print(f"  FPS = {fps:.2f}  (slow_factor={slow_factor})")

    use_ffmpeg = _has_ffmpeg()
    if not use_ffmpeg:
        print("WARNING: ffmpeg not found – falling back to OpenCV (lower quality)")

    videos_created = []
    for field_name, pngs in _discover_fields(run_dir):
        if fields and field_name not in fields:
            continue

        print(f"\n{'─'*60}")
        print(f"  {field_name}  ({len(pngs)} frames)")
        print(f"{'─'*60}")

        out_mp4 = run_dir / f"{field_name}.mp4"

        if use_ffmpeg:
            _make_video_ffmpeg(
                pngs, out_mp4, fps, dt, save_every, time_overlay, crf,
            )
        else:
            _make_video_opencv(pngs, out_mp4, fps, dt, save_every)

        if out_mp4.exists():
            size_mb = out_mp4.stat().st_size / 1e6
            print(f"  → {out_mp4}  ({size_mb:.1f} MB)")
            videos_created.append(out_mp4)

    print(f"\n{'═'*60}")
    print(f"  {len(videos_created)} video(s) created in {run_dir}")
    print(f"{'═'*60}")


def _make_video_ffmpeg(pngs, out_mp4, fps, dt, save_every, time_overlay, crf):
    """
    Use ffmpeg with the concat demuxer for frame-perfect, high-quality output.
    The concat approach handles non-sequential filenames correctly.
    """
    # Build a temporary concat file listing every frame with its duration
    concat_file = out_mp4.with_suffix(".concat.txt")
    frame_dur = 1.0 / fps
    with open(concat_file, "w") as f:
        for png in pngs:
            # ffmpeg concat expects paths with single-quote escaping
            escaped = str(png.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write(f"duration {frame_dur:.6f}\n")
        # Repeat last frame to avoid ffmpeg dropping it
        escaped = str(pngs[-1].resolve()).replace("'", "'\\''")
        f.write(f"file '{escaped}'\n")

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
    ]

    # pad to even dimensions (libx264 requires width & height divisible by 2)
    pad_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

    # Optional time overlay using drawtext filter
    if time_overlay and dt is not None and save_every is not None:
        # Build the time expression: t_sim = frame_number * save_every * dt
        dt_frame = save_every * dt
        # drawtext can use frame number via n
        drawtext = (
            f"drawtext=text='t = %{{eif\\:{dt_frame}*n\\:d\\:3}} s':"
            f"fontsize=28:fontcolor=black:"
            f"x=(w-text_w)/2:y=30:"
            f"borderw=2:bordercolor=white"
        )
        cmd += ["-vf", f"{pad_filter},{drawtext}"]
    else:
        cmd += ["-vf", pad_filter]

    cmd += [
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",   # compatibility
        "-movflags", "+faststart",
        str(out_mp4),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Clean up concat file
    concat_file.unlink(missing_ok=True)

    if result.returncode != 0:
        # If drawtext failed (missing font), retry without overlay
        if time_overlay and "drawtext" in result.stderr:
            print("  (drawtext filter failed – retrying without time overlay)")
            _make_video_ffmpeg(pngs, out_mp4, fps, dt, save_every, False, crf)
        else:
            print(f"  ffmpeg error:\n{result.stderr[-500:]}")


def _make_video_opencv(pngs, out_mp4, fps, dt, save_every):
    """Fallback: use OpenCV VideoWriter (no H.264 guarantee)."""
    import cv2

    frame0 = cv2.imread(str(pngs[0]))
    h, w = frame0.shape[:2]

    preferred_codecs = ["avc1", "H264", "X264", "mp4v"]
    writer = None
    for codec in preferred_codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        print("  ERROR: no compatible codec found")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, png in enumerate(pngs):
        frame = cv2.imread(str(png))
        if frame is None:
            continue
        if dt is not None and save_every is not None:
            iteration = _iteration_from_name(png)
            t_sim = iteration * dt
            cv2.putText(
                frame,
                f"t = {t_sim:.3f} s",
                (w // 2 - 80, 40),
                font, 0.8, (0, 0, 0), 2, cv2.LINE_AA,
            )
        writer.write(frame)

    writer.release()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MP4 videos from NS solver PNG output.",
    )
    parser.add_argument(
        "path",
        help="Timestamped run directory, or parent save_path (picks latest run).",
    )
    parser.add_argument(
        "--fields", nargs="*", default=None,
        help="Only generate videos for these sub-folders (e.g. omega_z_3d pressure_3d).",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override frame rate (default: computed from dt & save_every).",
    )
    parser.add_argument(
        "--slow-factor", type=float, default=1.0,
        help="Real-time multiplier (default: 1.0 = physical time).",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable time-stamp overlay.",
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="H.264 quality (default 18; lower = better).",
    )
    args = parser.parse_args()

    run_dir = Path(args.path)
    # If the user passed a parent save_path (no parameters.yaml), auto-detect latest
    if not (run_dir / "parameters.yaml").exists():
        run_dir = _latest_run(run_dir)
        print(f"Auto-selected latest run: {run_dir}")

    make_videos(
        run_dir,
        fields=args.fields,
        fps=args.fps,
        slow_factor=args.slow_factor,
        time_overlay=not args.no_overlay,
        crf=args.crf,
    )


if __name__ == "__main__":
    main()

