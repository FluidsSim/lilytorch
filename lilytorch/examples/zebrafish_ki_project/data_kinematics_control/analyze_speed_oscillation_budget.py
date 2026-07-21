"""Why does the EXPERIMENTAL speed trace oscillate so much more than the sim's?

THE ANSWER (2026-07-21), for the band that actually matters (20-45 Hz, where
the experiment swings 15.8 BL/s peak-to-peak against the sim's 1.44):

    **The tracked midline changes LENGTH, and a real fish's midline cannot.**

The tracked arc length breathes 3.87% rms / 19.9% peak-to-peak, individual
segments 5-8.5%.  No muscle contraction shortens a fish's body by a fifth --
this is a measurement effect (2D projection of a body that is not exactly in
the image plane, plus DLC labels sliding along the body).  The sim's tracked
chain is 16 RIGID links: its arc length is fixed by construction (0.149% rms,
26x less), so it structurally cannot produce the signal.

The metric is |d/dt (mean of tracked points)|.  A point set whose extent is
breathing moves its own centroid, and d/dt reads that out as "speed":

    in the 20-45 Hz band, corr(centroid speed, |dL/dt|) = 0.83
    |dL/dt| rms = 4.90 BL/s -> predicts ~2.45 BL/s of centroid speed
    observed centroid speed rms = 3.15 BL/s

FORWARD TEST (`inextensibility_test`): take the simulated swimming EXACTLY AS
IS and merely give its rigid midline the same per-segment length variation the
tracking has.  Nothing about the physics, gait, forces or coupling changes:

    centroid speed, 20-45 Hz          rms      peak-to-peak
      sim as simulated               0.31          1.44
      sim + tracking length variation 2.03        11.65
      REAL experiment                3.15         15.80
    full trace: sim std 3.09 -> 3.96 (real 4.41); cv 0.451 -> 0.562 (real 0.728)

That single artifact reproduces ~2/3 of the entire amplitude gap.  The residual
is a factor ~1.55, not ~10 -- and part of that is genuine: the real fish's
centroid-relative shape motion at 20-45 Hz is 2-3x the sim's (0.0158 vs 0.0052
BL at the head), i.e. real body-wave harmonic content the sim under-resolves.
THAT is the tractable physical residual worth chasing; the factor of 10 was not.

A momentum budget (below) independently bounds how much of the signal can be
translation at all, but note it leans on the sim's own measured force, so for
the 20-45 Hz band the inextensibility argument above is the stronger one --
it uses only the experimental data.

The compare metric is |d/dt (mean of tracked points)|.  If that centroid trace
were the body's CoM, then a narrow-band component of velocity amplitude v at
frequency f requires a net external force

    F = M * a = M * 2*pi*f * v .

So for a 50 mg, 18 mm zebrafish we can convert the MEASURED net hydrodynamic
force (drags.h5, per band) into the largest centroid surge that is physically
permitted at that frequency, and compare both sides against it.

    band [Hz]     REAL     SIM   F_meas uN  v_allowed  REAL/allowed
    2-5           1.18    0.57       7.0       0.39         3.0
    5-10          0.27    0.14       6.2       0.15         1.8
    10-20         0.53    0.13      29.6       0.37         1.4
    20-45         3.15    0.31      29.5       0.17        18.2
    45-100        1.35    0.07      17.1       0.04        30.2
    100-300       1.22    0.01       3.0       0.00        396.3
                                                    (speeds in BL/s rms)

Below 20 Hz -- the stroke band -- both sides sit within a small factor of the
bound: that part is real swimming, and the sim is NOT deficient there.  Above
20 Hz, where **72% of the experimental variance lives** (3.15 of 4.41 BL/s
total), the experimental trace exceeds the Newton bound by 18x, 30x and 396x.
The sim sits at or just under the bound in every band.  So the excess is a
property of the measured point set, not of the swimming, and no change to the
solver, gait, fins or coupling can or should reproduce it.

The method is validated on the sim, where the answer is known independently:
band-by-band, M*a of the true mass CoM reproduces the measured hydro force to a
few percent (14.6/25.0/27.9/19.9 uN implied vs 14.6/25.0/27.1/18.9 measured).

What the excess is NOT, and what it is:
  * NOT sample-to-sample tracking jitter.  A 2nd-difference estimator (immune
    to trend/leakage) puts the per-keypoint white-noise floor at 0.4 um =
    0.002 px -- the DLC data has already been smoothed.  The earlier "27% of
    real variance is >40 Hz DLC jitter" claim is wrong.
  * It IS smooth, sub-pixel wobble of the tracked midline at 20-300 Hz
    (~0.2 px per keypoint at 100-300 Hz, 20-30x the sim's), together with the
    fact that the tracked midline is EXTENSIBLE -- its arc length breathes
    3.87% rms / 19.9% peak-to-peak, per-segment 5-8.5% -- whereas the sim's
    tracked chain is a rigid-link chain whose arc length is fixed by
    construction (0.15% rms, 26x less).  The two sides feed the same metric
    structurally different objects.

CAVEAT: this bounds how much of the signal can be translation; it does not
fully attribute the remainder.  At 20-45 Hz the real fish's centroid-relative
shape motion is genuinely 2-3x the sim's (0.0158 vs 0.0052 BL at the head), so
part of that band may be real body-wave harmonic content the sim under-resolves.
A coherence test across keypoints did NOT separate artifact from body wave
(real and sim are equally coherent, 0.29-0.44, in every band above 20 Hz).

Usage
-----
    python analyze_speed_oscillation_budget.py [sim_run_dir]
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

BL = 0.018        # body length [m]
MASS = 5.022e-5   # total body mass [kg], from simulation.hdf5 /links/masses
NS_DATA = "/data/andreaferrario/ns_data"
KEYPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keypoints")
REAL_CSV = os.path.join(KEYPOINTS, "ep248_Cl2_slow_fish13_XY_BL.csv")
OUT_PNG = os.path.join(NS_DATA, "speed_oscillation_newton_budget.png")

BANDS = [(2, 5), (5, 10), (10, 20), (20, 45), (45, 100), (100, 300)]


def bandpass(sig, fs, lo, hi, order=4):
    """Zero-phase band-pass.  SOS form: a 4th-order Butterworth in transfer-
    function form goes numerically unstable at these normalised cutoffs, and an
    FFT brick-wall leaks the translation ramp into every band."""
    sos = butter(order, [lo / (fs / 2), min(hi, 0.45 * fs) / (fs / 2)],
                 btype="band", output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def load_real():
    d = pd.read_csv(REAL_CSV)
    t = d["time_ms"].to_numpy() / 1000.0
    x = d[[c for c in d.columns if c.startswith("x")]].to_numpy()
    y = d[[c for c in d.columns if c.startswith("y")]].to_numpy()
    fs = 1.0 / np.mean(np.diff(t))
    speed = np.hypot(np.gradient(x.mean(1), t), np.gradient(y.mean(1), t))
    return speed, fs, x, y


def load_sim(run_dir):
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        pos = f["FARMSLISTanimats/0/sensors/links/array"][:][:, :, 0:3]
        ts = f["times"][:]
    fs = 1.0 / np.mean(np.diff(ts))
    cx, cy = pos[:, :, 0] / BL, pos[:, :, 1] / BL
    speed = np.hypot(np.gradient(cx.mean(1), ts), np.gradient(cy.mean(1), ts))
    with h5py.File(os.path.join(run_dir, "drags.h5"), "r") as f:
        force = (f["pressure_drags"][:] + f["viscous_drags"][:]).transpose(2, 0, 1).sum(1)
    return speed, fs, force, pos


def inextensibility_test(kx, ky, fs_real, pos, ts, lo=20, hi=45):
    """Give the sim's RIGID midline the tracking's own per-segment length
    variation, change nothing else, and re-measure the metric."""
    fs_sim = 1.0 / np.mean(np.diff(ts))
    seg_r = np.hypot(np.diff(kx, axis=1), np.diff(ky, axis=1))
    ratio = seg_r / seg_r.mean(0)                       # per-segment modulation
    t_real = np.arange(len(kx)) / fs_real

    P = pos[:, :, :2] / BL
    v = np.diff(P, axis=1)
    l = np.linalg.norm(v, axis=2)
    u = v / l[:, :, None]
    ns = l.shape[1]

    # map the real 9-segment profile onto the sim's 15 segments, then onto ts
    ur = (np.arange(ratio.shape[1]) + 0.5) / ratio.shape[1]
    us = (np.arange(ns) + 0.5) / ns
    # ratio is (T_real, 9); resample along the body, then along time
    ratio_sim_segs = np.vstack([np.interp(us, ur, ratio[i]) for i in range(len(t_real))])
    R = np.vstack([np.interp(ts, t_real, ratio_sim_segs[:, j]) for j in range(ns)]).T
    Pm = np.empty_like(P)
    Pm[:, 0] = P[:, 0]
    for i in range(ns):
        Pm[:, i + 1] = Pm[:, i] + u[:, i] * (l[:, i] * R[:, i])[:, None]

    def speed(Q):
        c = Q.mean(1)
        return np.hypot(np.gradient(c[:, 0], ts), np.gradient(c[:, 1], ts))

    s0, s1 = speed(P), speed(Pm)
    sr = np.hypot(np.gradient(kx.mean(1), t_real), np.gradient(ky.mean(1), t_real))

    arc_r = seg_r.sum(1)
    print(f"\ntracked arc length: REAL {100*arc_r.std()/arc_r.mean():.2f}% rms, "
          f"{100*np.ptp(arc_r)/arc_r.mean():.1f}% p2p  --  a fish midline is INEXTENSIBLE")
    dL = np.gradient(arc_r, t_real)
    print(f"  in {lo}-{hi} Hz: |dL/dt| rms {bandpass(dL, fs_real, lo, hi).std():.2f} BL/s,"
          f"  corr(centroid speed, |dL/dt|) = "
          f"{np.corrcoef(np.abs(bandpass(sr,fs_real,lo,hi)), np.abs(bandpass(dL,fs_real,lo,hi)))[0,1]:.2f}")
    print(f"\n  FORWARD TEST -- centroid speed, {lo}-{hi} Hz    {'rms':>8} {'p2p':>8}")
    for lbl, b in [("sim as simulated (rigid chain)", bandpass(s0, fs_sim, lo, hi)),
                   ("sim + tracking length variation", bandpass(s1, fs_sim, lo, hi)),
                   ("REAL experiment", bandpass(sr, fs_real, lo, hi))]:
        print(f"    {lbl:<38} {b.std():8.2f} {np.ptp(b):8.2f} BL/s")
    print(f"    full trace: sim std {s0.std():.2f} -> {s1.std():.2f} (real {sr.std():.2f});"
          f" cv {s0.std()/s0.mean():.3f} -> {s1.std()/s1.mean():.3f} (real {sr.std()/sr.mean():.3f})")


def main(argv):
    run_dir = (argv[1] if len(argv) > 1 else "2026-07-20T12:17:59.330717")
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(NS_DATA, run_dir)

    s_real, fs_real, kx, ky = load_real()
    s_sim, fs_sim, force, pos = load_sim(run_dir)

    print(f"real centroid speed: mean {s_real.mean():.2f} std {s_real.std():.2f} "
          f"cv {s_real.std()/s_real.mean():.3f} BL/s")
    print(f"sim  centroid speed: mean {s_sim.mean():.2f} std {s_sim.std():.2f} "
          f"cv {s_sim.std()/s_sim.mean():.3f} BL/s\n")

    print(f"  {'band [Hz]':<11} {'REAL':>8} {'SIM':>8} {'F_meas uN':>11}"
          f" {'v_allowed':>9} {'REAL/allow':>11}")
    print("  " + "-" * 62)
    rows = []
    for lo, hi in BANDS:
        fc = np.sqrt(lo * hi)                       # geometric band centre
        a = bandpass(s_real, fs_real, lo, hi).std()
        b = bandpass(s_sim, fs_sim, lo, hi).std()
        fb = np.hypot(bandpass(force[:, 0], fs_sim, lo, hi),
                      bandpass(force[:, 1], fs_sim, lo, hi)).std()
        allowed = fb / (MASS * 2 * np.pi * fc) / BL
        rows.append((f"{lo}-{hi}", a, b, fb * 1e6, allowed))
        print(f"  {lo}-{hi:<9} {a:8.2f} {b:8.2f} {fb*1e6:11.1f}"
              f" {allowed:9.2f} {a/allowed:11.1f}")

    # arc-length extensibility of the two tracked point sets
    seg_r = np.hypot(np.diff(kx, axis=1), np.diff(ky, axis=1)).sum(1)
    seg_s = np.linalg.norm(np.diff(pos, axis=1), axis=2).sum(1) / BL
    print(f"\ntracked-midline arc length: REAL {100*seg_r.std()/seg_r.mean():.2f}% rms "
          f"({100*(seg_r.max()-seg_r.min())/seg_r.mean():.1f}% p2p)"
          f"  vs  SIM {100*seg_s.std()/seg_s.mean():.3f}% (rigid-link chain)")

    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        ts = f["times"][:]
    inextensibility_test(kx, ky, fs_real, pos, ts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(len(rows))
    w = 0.27
    ax.bar(idx - w, [r[1] for r in rows], w, label="experiment", color="k")
    ax.bar(idx, [r[2] for r in rows], w, label="sim", color="tab:blue")
    ax.bar(idx + w, [r[4] for r in rows], w,
           label="max allowed by Newton\n(from measured hydro force)",
           color="tab:red", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlabel("frequency band [Hz]")
    ax.set_ylabel("centroid-speed rms [BL/s]")
    ax.set_title("Above 20 Hz the experimental speed exceeds what any force could produce")
    ax.axvline(2.5, color="0.5", ls="--", lw=1)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y * 6)          # headroom so the labels clear the bars
    ax.text(1.0, hi_y * 2.2, "real swimming:\nboth sides near the bound",
            ha="center", va="center", color="0.3", fontsize=9)
    ax.text(4.0, hi_y * 2.2, "experiment 18-396x over the bound:\nnot CoM motion",
            ha="center", va="center", color="tab:red", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print("\nwrote", OUT_PNG)


if __name__ == "__main__":
    main(sys.argv)
