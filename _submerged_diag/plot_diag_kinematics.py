"""Standard forward/lateral swim-speed plot for the submerged-diag runs.

Reuses the plot_vels_ms004mpt003.py methodology (best-fit swim plane -> 4 Hz
Butterworth low-pass -> quadratic-trajectory tangent projection) on the head
(link0) trajectory logged by speed_logger.py, and overlays the experiment
forward speed.  One PNG per run, written to the standard ns_data location.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/data/andreaferrario/ns_data/diag_speed_plots"
EXP_TRACK = "/data/andreaferrario/1guilla_experiments/swim/videos/ms004mpt003_track.csv"
ROBOT_FWD = 0.128


def _lp(a, fps, cut=4.0):
    valid = ~np.isnan(a)
    if valid.sum() < 10:
        return a
    wn = cut / (0.5 * fps)
    if wn >= 1.0:
        return a
    b, c = butter(4, wn, btype="low")
    return filtfilt(b, c, np.interp(np.arange(len(a)), np.flatnonzero(valid), a[valid]))


def _plane(p3):
    c = p3.mean(0); pc = p3 - c
    _, _, vh = np.linalg.svd(pc, full_matrices=False)
    pp = pc @ vh[:2].T
    if pp[-1, 0] - pp[0, 0] < 0:
        pp[:, 0] *= -1
    return pp - pp[0]


def _project(t, x, y):
    px = np.polyfit(t, x, 2); py = np.polyfit(t, y, 2)
    vx = np.gradient(x, t); vy = np.gradient(y, t)
    tx = 2 * px[0] * t + px[1]; ty = 2 * py[0] * t + py[1]
    n = np.hypot(tx, ty); n = np.where(n > 0, n, 1.0)
    txu, tyu = tx / n, ty / n
    fwd = vx * txu + vy * tyu
    lat = vx * (-tyu) + vy * txu
    return fwd, lat


def run(tag, csv):
    d = np.genfromtxt(csv, delimiter=",", names=True)
    t = d["t"]
    p3 = np.column_stack([d["x"], d["y"], d["z"]])
    pp = _plane(p3)
    fps = 1.0 / np.median(np.diff(t))
    xs, ys = _lp(pp[:, 0], fps), _lp(pp[:, 1], fps)
    fwd, lat = _project(t, xs, ys)
    m = t >= 1.5  # drop the startup transient for the mean
    mean_fwd = float(np.mean(fwd[m]))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, fwd, color="tab:orange", lw=2, label=f"forward (sim), mean={mean_fwd:.3f} m/s")
    ax.plot(t, lat, color="tab:green", lw=1.5, alpha=0.8, label="lateral (sim)")
    ax.axhline(mean_fwd, color="tab:orange", ls=":", lw=1.2)
    ax.axhline(ROBOT_FWD, color="k", ls="--", lw=1.5, label=f"robot fwd = {ROBOT_FWD} m/s")
    if os.path.exists(EXP_TRACK):
        e = pd.read_csv(EXP_TRACK)
        ax.plot(e["time_s"], e["speed_fwd_mps"], color="tab:orange", ls="--",
                lw=1.3, alpha=0.6, label="forward (exp)")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]")
    ax.set_title(f"{tag}  (TRANSVERSE, frelax=0.5)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"speed_{tag}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"{tag:<22} mean forward = {mean_fwd:.3f} m/s   -> {out}")


if __name__ == "__main__":
    runs = sys.argv[1:] or ["surf_T", "sub_T", "sp_T"]
    for tag in runs:
        run(tag, f"/data/andreaferrario/lilytorch/_submerged_diag/speed_{tag}.csv")
