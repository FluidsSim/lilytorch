"""Report zebrafish swim speed per readout case, from simulation.hdf5.

The symptom that started all of this is "the fish swims markedly slower under
eulerian", so this measures exactly that: net forward COM displacement and mean
forward speed over each run in the arbitration stack.

    python -m lilytorch.examples.force_benchmarks.zfish_swim_speed

Forward is -x for this gait (the animat is spawned yaw=pi).
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

STACK = "/data/andreaferrario/ns_data/zfish_readout_arbitration"


def _latest(case_dir):
    subs = sorted(d for d in glob.glob(os.path.join(case_dir, "*")) if os.path.isdir(d))
    for d in reversed(subs):
        if os.path.exists(os.path.join(d, "output", "simulation.hdf5")):
            return d
    return None


def speed(run_dir):
    from farms_core.sensors.sensor_convention import sc
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        link = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        masses = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["masses"])
        dt = float(np.array(f["timestep"]))
    # mass-weighted COM over links, x/y position channels
    px = link[:, :, sc.link_urdf_position_x]
    py = link[:, :, sc.link_urdf_position_y]
    w = masses / masses.sum()
    com_x = (px * w[None, :]).sum(1)
    com_y = (py * w[None, :]).sum(1)
    t = dt * np.arange(len(com_x))
    dx = com_x - com_x[0]
    dy = com_y - com_y[0]
    dist = np.hypot(dx, dy)
    return dict(t=t, com_x=com_x, dx=dx[-1], dy=dy[-1], dist=dist[-1],
                v_mean=dist[-1] / max(t[-1], 1e-12), T=t[-1])


def main(stack=STACK):
    cases = sorted(d for d in glob.glob(os.path.join(stack, "*")) if os.path.isdir(d))
    if not cases:
        raise SystemExit(f"no cases under {stack}")
    print(f"{'case':>22} {'T [s]':>7} {'dx [mm]':>9} {'dy [mm]':>9} "
          f"{'|d| [mm]':>9} {'v [mm/s]':>9} {'v/BL/s':>7}")
    BL = 0.018            # ~18 mm larva
    rows = {}
    for c in cases:
        rd = _latest(c)
        if rd is None:
            print(f"{os.path.basename(c):>22}   (no completed run)")
            continue
        s = speed(rd)
        rows[os.path.basename(c)] = s
        print(f"{os.path.basename(c):>22} {s['T']:7.3f} {s['dx']*1e3:9.3f} "
              f"{s['dy']*1e3:9.3f} {s['dist']*1e3:9.3f} {s['v_mean']*1e3:9.3f} "
              f"{s['v_mean']/BL:7.2f}")
    ref = rows.get("lagr_off0")
    if ref:
        print(f"\nratios of mean speed vs lagr_off0 ({ref['v_mean']*1e3:.3f} mm/s):")
        for k, v in rows.items():
            print(f"  {k:>22}  {v['v_mean']/ref['v_mean']:.3f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else STACK)
