"""Quantify the arena-wall contribution to the zebrafish sim's residual turn.

Background (2026-07-21).  ``compare_sim_real.py`` showed the sim turning ~+53 deg
over a 0.55 s episode while the real ep248 fish swam straight.  Two sessions
ruled out the gait, the keypoint->angle conversion, PD tracking, the coupling
scheme and the force readout.  What was never checked is the ARENA: the
production config puts an 18 mm fish in a 0.10 x 0.05 m box (2.8 BL wide) and
spawns it at yaw 134.93 deg, i.e. heading -45 deg, aimed diagonally at the ymin
wall.  Lateral wall clearance therefore decays through the episode, and the
turn tracks it.

This script recomputes, for any set of run folders:

  * lateral wall clearance  = min over links of the distance to the ymin/ymax
    wall faces, in body lengths.  The pool walls are boxes placed OUTSIDE the
    fluid box (gen_pool_sdf puts wall_ymin at ymin - wt/2 with thickness wt),
    so the inner faces sit exactly at ymin / ymax.
  * heading                 = angle of the link0 (head) - link15 (tail) chord,
    smoothed over one tail beat (~125 steps at 32 Hz, dt = 0.25 ms).
  * turn rate               = d(heading)/dt.

and pools the samples into clearance bins.  The signature of a wall-driven turn
is a turn rate that is ~zero at large clearance and rises steeply below ~1 BL.

OUTCOME (2026-07-21): on the 2026-07-20 runs this binning looks damning
(+5.9 deg/s above 0.8 BL, +87..+310 deg/s below), but it is a TIME CONFOUND --
clearance decays monotonically in every one of those runs, so "low clearance"
merely means "late in the episode".  The open-water control run
(gen_configs_pd_3d_ep248_slow_openwater.py) holds >1.2 BL clearance and turns
exactly like the baseline, so **confinement is NOT the driver**.  Walls do bend
the heading, but only inside ~0.2 BL clearance.  Keep this script for re-checking
confinement if the arena is ever shrunk -- and always include a run whose
clearance does NOT decay with time before reading anything into the bins.

Usage
-----
    python analyze_wall_effect.py [run_dir ...]

With no arguments it uses the 2026-07-20 baseline set.
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np

BL = 0.018       # zebrafish body length [m] (link0-link15 chord 0.0152 + tips)
DT = 0.00025     # solver timestep [s]
BEAT = 125       # steps per tail beat (~32 Hz) -> smoothing window

# Fluid-box faces; pool walls sit immediately outside these.
YMIN, YMAX = -0.025, 0.025

DEFAULT_RUNS = [
    "2026-07-20T11:27:36.133722",
    "2026-07-20T12:17:59.330717",
    "2026-07-20T18:10:10.765745",
    "2026-07-20T21:42:11.244801",
    "2026-07-20T21:47:10.073694",
    "zebrafish_real_kinematics/slow",
]
NS_DATA = "/data/andreaferrario/ns_data"


def load(run_dir):
    """Return (clearance_BL, smoothed_heading_deg, turn_rate_deg_s) for *run_dir*.

    Trailing rows are dropped where the episode crashed or the record buffer was
    never filled (time stops increasing, or positions are zero/non-finite).
    """
    path = os.path.join(run_dir, "output", "simulation.hdf5")
    with h5py.File(path, "r") as f:
        links = f["FARMSLISTanimats/0/sensors/links/array"][:]
        times = f["times"][:]

    pos = links[:, :, 0:3]
    n = min(len(times), len(pos))
    times, pos = times[:n], pos[:n]

    ok = (
        np.r_[True, np.diff(times) > 0]
        & np.isfinite(pos).all((1, 2))
        & (np.abs(pos).sum((1, 2)) > 1e-9)
    )
    n = int(np.argmin(ok)) if (~ok).any() else n
    pos = pos[:n]
    if n < 600:
        return None

    chord = pos[:, 0, :2] - pos[:, -1, :2]
    heading = np.degrees(np.unwrap(np.arctan2(chord[:, 1], chord[:, 0])))
    heading = np.convolve(heading, np.ones(BEAT) / BEAT, mode="same")

    clearance = np.minimum(pos[:, :, 1] - YMIN, YMAX - pos[:, :, 1]).min(1) / BL
    return clearance, heading, np.gradient(heading, DT)


def main(argv):
    runs = argv[1:] or DEFAULT_RUNS
    pooled_clear, pooled_rate = [], []

    print(f"{'run':<42} {'n':>5}  clearance BL      net turn")
    for run in runs:
        run_dir = run if os.path.isabs(run) else os.path.join(NS_DATA, run)
        got = load(run_dir)
        if got is None:
            print(f"{os.path.basename(run_dir.rstrip('/')):<42} crashed too early, skipped")
            continue
        clearance, heading, rate = got
        n = len(clearance)
        keep = slice(BEAT + 25, n - BEAT - 25)   # drop the convolution edges
        pooled_clear.append(clearance[keep])
        pooled_rate.append(rate[keep])
        print(
            f"{os.path.basename(run_dir.rstrip('/')):<42} {n:5d}  "
            f"{clearance[keep][0]:.2f} -> {clearance[keep][-1]:.2f}   "
            f"{heading[keep][-1] - heading[keep][0]:+6.0f} deg"
        )

    clearance = np.concatenate(pooled_clear)
    rate = np.concatenate(pooled_rate)

    print(f"\npooled over {len(runs)} runs, {len(clearance)} samples")
    print(" clearance (BL)      n    mean turn rate (deg/s)")
    for lo, hi in [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .8), (.8, 1.5)]:
        m = (clearance >= lo) & (clearance < hi)
        if m.sum() > 50:
            print(f"  {lo:.1f} - {hi:.1f}      {m.sum():5d}      {rate[m].mean():+8.1f}")


if __name__ == "__main__":
    main(sys.argv)
