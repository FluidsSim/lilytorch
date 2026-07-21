"""Where, in time, does the sim's heading stop following the real fish's?

Background (2026-07-21).  The sim's residual turn was assumed to be a steady
curvature bias integrated by an open-loop body.  Plotting the BEAT-SMOOTHED
heading of both sides against time shows something different: the real ep248
fish does not swim straight at all -- it executes a slow yaw oscillation of
about +-13 deg (dip to -13 deg at t = 0.10-0.15 s, back up through zero, peak
+11 deg at t = 0.35 s, then back toward zero).  The sim REPRODUCES that
manoeuvre closely for the first ~0.30 s and only then diverges: the real
heading returns toward zero while the sim's keeps ratcheting up to +43 deg.

So the defect is not a bias that gets integrated; it is a failure to ARREST a
low-frequency yaw excursion that the real fish arrests.  That points at yaw
restoring/damping (fin area, closed-loop heading control), and it also means
the hydrodynamics and coupling are in better shape than assumed -- they
reproduce a non-trivial real yaw manoeuvre for ~5 tail beats.

Both sides use the same metric: the head-tail chord angle, unwrapped and
smoothed over one tail beat (~32 Hz), referenced to its own value at t = 0.02 s.

Usage
-----
    python analyze_heading_divergence.py [sim_run_dir ...]
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pandas as pd

KEYPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keypoints")
REAL_CSV = os.path.join(KEYPOINTS, "ep248_Cl2_slow_fish13_XY_BL.csv")
NS_DATA = "/data/andreaferrario/ns_data"
OUT_PNG = os.path.join(NS_DATA, "heading_divergence_sim_vs_real.png")

BEAT_HZ = 32.0
T_REF = 0.02


def _smooth_heading(head_xy, tail_xy, fs):
    """Beat-smoothed, unwrapped chord angle in degrees, referenced to T_REF."""
    ang = np.degrees(np.unwrap(np.arctan2(head_xy[:, 1] - tail_xy[:, 1],
                                          head_xy[:, 0] - tail_xy[:, 0])))
    w = max(3, int(round(fs / BEAT_HZ)))
    sm = np.convolve(ang, np.ones(w) / w, mode="same")
    sm = sm - sm[int(T_REF * fs)]
    # 'same' convolution tapers against zero-padding at both ends; blank the
    # half-window that is not a true beat average.
    sm[: w // 2] = np.nan
    sm[len(sm) - w // 2:] = np.nan
    return sm


def load_real():
    d = pd.read_csv(REAL_CSV)
    t = d["time_ms"].to_numpy() / 1000.0
    x = d[[c for c in d.columns if c.startswith("x")]].to_numpy()
    y = d[[c for c in d.columns if c.startswith("y")]].to_numpy()
    fs = 1.0 / np.mean(np.diff(t))
    return t, _smooth_heading(np.c_[x[:, 0], y[:, 0]], np.c_[x[:, -1], y[:, -1]], fs)


def load_sim(run_dir):
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        links = f["FARMSLISTanimats/0/sensors/links/array"][:]
        times = f["times"][:]
    pos = links[:, :, 0:3]
    n = min(len(times), len(pos))
    times, pos = times[:n], pos[:n]
    ok = (np.r_[True, np.diff(times) > 0]
          & np.isfinite(pos).all((1, 2))
          & (np.abs(pos).sum((1, 2)) > 1e-9))
    n = int(np.argmin(ok)) if (~ok).any() else n
    times, pos = times[:n], pos[:n]
    fs = 1.0 / np.mean(np.diff(times))
    return times, _smooth_heading(pos[:, 0, :2], pos[:, -1, :2], fs)


def main(argv):
    runs = argv[1:] or ["2026-07-20T12:17:59.330717", "2026-07-21T10:00:19.717441"]
    t_real, h_real = load_real()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_real, h_real, "k", lw=2.5, label="real ep248 (keypoints)")

    grid = np.arange(0.03, 0.53, 0.01)
    print(f"{'run':<30} {'RMS err 0.02-0.30s':>19} {'RMS err 0.30-0.52s':>20}")
    real_on_grid = np.interp(grid, t_real, h_real)

    for run in runs:
        run_dir = run if os.path.isabs(run) else os.path.join(NS_DATA, run)
        t_sim, h_sim = load_sim(run_dir)
        label = os.path.basename(run_dir.rstrip("/"))
        ax.plot(t_sim, h_sim, lw=1.5, label=f"sim {label[:19]}")

        sim_on_grid = np.interp(grid, t_sim, h_sim)
        err = sim_on_grid - real_on_grid
        early = grid < 0.30
        # runs that crashed early are NaN past their last beat average
        print(f"{label[:30]:<30} {np.sqrt(np.nanmean(err[early]**2)):>16.1f} deg"
              f" {np.sqrt(np.nanmean(err[~early]**2)):>17.1f} deg")

    ax.axvspan(0.02, 0.30, color="tab:green", alpha=0.08)
    ax.axvspan(0.30, 0.55, color="tab:red", alpha=0.08)
    ax.text(0.15, -20, "sim tracks real", ha="center", color="tab:green")
    ax.text(0.42, -20, "sim fails to return", ha="center", color="tab:red")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("beat-smoothed heading change [deg]")
    ax.set_title("Sim reproduces the real yaw manoeuvre for ~0.3 s, then fails to arrest it")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print("\nwrote", OUT_PNG)


if __name__ == "__main__":
    main(sys.argv)
