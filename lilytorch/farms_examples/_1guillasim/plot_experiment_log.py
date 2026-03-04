"""
Script to read and plot desired vs actual joint positions
from 1Guilla robot experiment log files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────
LOG_DIR = "/data/andreaferrario/1guilla_experiment_dye/1guilla_videos_swim_water/log"
SAVE_DIR = "/data/andreaferrario/lilytorch/figures/experiment_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


def read_experiment_csv(filepath):
    """
    Read a 1Guilla experiment log CSV.

    Structure:
      Row 0 : metadata header  (NumMotors, positionAmplitude[deg], Lambda, Frequency[Hz])
      Row 1 : metadata values
      Row 2 : column headers for the time-series data
      Row 3+: data
    """
    # ── Read metadata ──────────────────────────────────────────────────
    meta_df = pd.read_csv(filepath, nrows=1, header=0)
    metadata = {
        "NumMotors": int(meta_df["NumMotors"].iloc[0]),
        "positionAmplitude_deg": float(meta_df["positionAmplitude[deg]"].iloc[0]),
        "Lambda": float(meta_df["Lambda"].iloc[0]),
        "Frequency_Hz": float(meta_df["Frequency[Hz]"].iloc[0]),
    }

    # ── Read data (skip the 2 metadata rows) ───────────────────────────
    data_df = pd.read_csv(filepath, skiprows=2, header=0)

    # Build a time vector from the two timestamp columns [s] and [us]
    ts_s = data_df.iloc[:, 0].values.astype(np.float64)
    ts_us = data_df.iloc[:, 1].values.astype(np.float64)
    t = ts_s + ts_us * 1e-6
    t = t - t[0]  # start from 0

    n_motors = metadata["NumMotors"]

    # Goal (desired) positions  – columns 2 .. 2+n_motors
    goal_cols = [f"{i}GoalPosition[rad]" for i in range(n_motors)]
    goal = data_df[goal_cols].values  # (N, n_motors)

    # Feedback (actual) positions – next n_motors columns
    fbck_cols = [f"{i}FbckPosition[rad]" for i in range(n_motors)]
    fbck = data_df[fbck_cols].values  # (N, n_motors)

    return t, goal, fbck, metadata, data_df


def plot_desired_vs_actual(t, goal, fbck, metadata, title_extra="", save_path=None):
    """Plot desired and actual joint positions for all motors."""
    n_motors = metadata["NumMotors"]
    fig, axes = plt.subplots(n_motors, 1, figsize=(14, 2.2 * n_motors), sharex=True)
    if n_motors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, np.rad2deg(goal[:, i]), label="Desired", linewidth=1.2)
        ax.plot(t, np.rad2deg(fbck[:, i]), label="Actual", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"Joint {i} [deg]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    freq = metadata["Frequency_Hz"]
    amp = metadata["positionAmplitude_deg"]
    lam = metadata["Lambda"]
    fig.suptitle(
        f"Desired vs Actual Joint Positions  "
        f"(f={freq} Hz, A={amp}°, λ={lam})  {title_extra}",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_tracking_error(t, goal, fbck, metadata, title_extra="", save_path=None):
    """Plot the tracking error (desired – actual) for all motors."""
    n_motors = metadata["NumMotors"]
    error = np.rad2deg(goal - fbck)

    fig, axes = plt.subplots(n_motors, 1, figsize=(14, 2.2 * n_motors), sharex=True)
    if n_motors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, error[:, i], linewidth=1.0, color="tab:red")
        ax.set_ylabel(f"Err {i} [deg]")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    freq = metadata["Frequency_Hz"]
    amp = metadata["positionAmplitude_deg"]
    lam = metadata["Lambda"]
    fig.suptitle(
        f"Tracking Error (Desired − Actual)  "
        f"(f={freq} Hz, A={amp}°, λ={lam})  {title_extra}",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_files = sorted(
        [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]
    )
    print(f"Found {len(csv_files)} CSV files in {LOG_DIR}\n")

    for fname in csv_files:
        fpath = os.path.join(LOG_DIR, fname)
        tag = fname.replace("log.csv", "")
        print(f"Processing {fname} …")

        t, goal, fbck, meta, _ = read_experiment_csv(fpath)

        plot_desired_vs_actual(
            t, goal, fbck, meta,
            title_extra=tag,
            save_path=os.path.join(SAVE_DIR, f"{tag}_desired_vs_actual.png"),
        )
        plot_tracking_error(
            t, goal, fbck, meta,
            title_extra=tag,
            save_path=os.path.join(SAVE_DIR, f"{tag}_tracking_error.png"),
        )

    print("\nDone.")
