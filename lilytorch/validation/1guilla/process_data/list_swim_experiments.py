"""
Script to build a summary CSV for the 1Guilla swim experiments.

Each video in VIDEO_DIR has a matching log file in LOG_DIR with the same
tag (e.g. ms001mpt001.mp4 ↔ ms001mpt001log.csv).  The summary contains:
  - video_filename
  - log_filename
  - positionAmplitude_deg
  - Lambda
  - Frequency_Hz
"""

import os
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────
MAIN_DIR  = "/data/andreaferrario/1guilla_experiments/swim"
VIDEO_DIR = MAIN_DIR + "/videos"
LOG_DIR   = MAIN_DIR + "/log"
SAVE_DIR  = MAIN_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def read_metadata(log_path):
    """Return amplitude, lambda and frequency from the first two rows of a log CSV."""
    meta_df = pd.read_csv(log_path, nrows=1, header=0)
    return {
        "positionAmplitude_deg": float(meta_df["positionAmplitude[deg]"].iloc[0]),
        "Lambda":                float(meta_df["Lambda"].iloc[0]),
        "Frequency_Hz":          float(meta_df["Frequency[Hz]"].iloc[0]),
    }


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    video_files = sorted(
        f for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    print(f"Found {len(video_files)} video files in {VIDEO_DIR}\n")

    rows = []
    for vname in video_files:
        stem     = os.path.splitext(vname)[0]          # e.g. ms001mpt001
        log_name = f"{stem}log.csv"
        log_path = os.path.join(LOG_DIR, log_name)

        if not os.path.isfile(log_path):
            print(f"  WARNING: log not found for {vname} (expected {log_name})")
            continue

        meta = read_metadata(log_path)
        rows.append({
            "video_filename":        vname,
            "log_filename":          log_name,
            "positionAmplitude_deg": meta["positionAmplitude_deg"],
            "Lambda":                meta["Lambda"],
            "Frequency_Hz":          meta["Frequency_Hz"],
        })
        print(f"  {vname}  A={meta['positionAmplitude_deg']}°  λ={meta['Lambda']}  f={meta['Frequency_Hz']} Hz")

    summary_df   = pd.DataFrame(rows)
    summary_path = os.path.join(SAVE_DIR, "swim_experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved → {summary_path}")
    print("Done.")
