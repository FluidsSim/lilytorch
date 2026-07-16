"""Arbitrate the force readouts by hydrodynamic-power closure.

THIS is the test that says which readout is RIGHT (not merely which disagree).

    P_BDIM = -Sum_links [ F_i . v_i + tau_i . omega_i ]        (drags.h5)
    dE_k/dt = P_BDIM - dissipation_rate                        (diagnostics.h5)

The left side is built from the READOUT forces.  The right side is measured from
the fluid fields alone and knows nothing about the readout.  So a readout that
over-reports drag claims to have pumped more power into the water than the water
actually received, and its closure ratio

    closure = Int P_BDIM dt / (dE_k + E_diss)

overshoots 100%.  Reference: the freq-cross study closed at 90-102% per case.

NOTE the minus sign is essential (Newton's 3rd law: F.v is power fluid->body).
NOTE the actuator power P_act is NOT the arbiter -- it is ~5000-12000x the power
that reaches the fluid (only ~0.02% of muscle work enters the water), so
verify_energy_balance.py's residual is dominated by internal body dynamics and
is identical for every readout.  Use this instead.

    python -m lilytorch.validation.force_readout_oracle.zfish_pbdim_closure
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

STACK = "/data/andreaferrario/ns_data/zfish_readout_arbitration"
TMAX = 0.3          # the clean, wall-free window for this gait
RHO = 1000.0


def _latest(case_dir):
    for d in reversed(sorted(glob.glob(os.path.join(case_dir, "*")))):
        if os.path.exists(os.path.join(d, "drags.h5")):
            return d
    return None


def closure(run_dir, tmax=TMAX):
    from farms_core.sensors.sensor_convention import sc

    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        link = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        dt = float(np.array(f["timestep"]))
    with h5py.File(os.path.join(run_dir, "drags.h5"), "r") as f:
        fp = np.array(f["pressure_drags"])        # (n_links, 3, nt)
        fv = np.array(f["viscous_drags"])
        tp = np.array(f["pressure_torques"])
        tv = np.array(f["viscous_torques"])
    with h5py.File(os.path.join(run_dir, "diagnostics.h5"), "r") as f:
        ke_raw = np.array(f["kinetic_energy"])
        diss_raw = np.array(f["dissipation_rate"])

    F = fp + fv                                   # total force ON body from fluid
    T = tp + tv
    nt = min(F.shape[2], link.shape[0])
    t = dt * np.arange(nt)

    # link COM linear / angular velocity, (nt, n_links, 3)
    v = np.stack([link[:nt, :, sc.link_com_velocity_lin_x],
                  link[:nt, :, sc.link_com_velocity_lin_y],
                  link[:nt, :, sc.link_com_velocity_lin_z]], axis=-1)
    w = np.stack([link[:nt, :, sc.link_com_velocity_ang_x],
                  link[:nt, :, sc.link_com_velocity_ang_y],
                  link[:nt, :, sc.link_com_velocity_ang_z]], axis=-1)
    Ft = np.transpose(F[:, :, :nt], (2, 0, 1))    # -> (nt, n_links, 3)
    Tt = np.transpose(T[:, :, :nt], (2, 0, 1))

    # minus: F.v is power fluid->body; we want body->fluid
    P_bdim = -((Ft * v).sum(-1) + (Tt * w).sum(-1)).sum(-1)     # (nt,)

    ke_valid = np.isfinite(ke_raw)
    ke_t = np.where(ke_valid)[0] * dt
    ke_J = RHO * ke_raw[ke_valid]
    diss_valid = np.isfinite(diss_raw)
    diss_t = np.where(diss_valid)[0] * dt
    diss_W = RHO * diss_raw[diss_valid]

    keep = ke_t <= tmax
    ke_t, ke_J = ke_t[keep], ke_J[keep]
    kd = diss_t <= tmax
    diss_t, diss_W = diss_t[kd], diss_W[kd]
    kp = t <= tmax
    t, P_bdim = t[kp], P_bdim[kp]

    # integrated form: Int P_BDIM dt  ==  dE_k + E_diss
    E_bdim = np.trapz(P_bdim, t)
    E_diss = np.trapz(diss_W, diss_t)
    dEk = ke_J[-1] - ke_J[0]
    received = dEk + E_diss
    return dict(P_bdim_mean=float(P_bdim.mean()), E_bdim=float(E_bdim),
                E_diss=float(E_diss), dEk=float(dEk), received=float(received),
                closure=float(E_bdim / received) if received else np.nan)


def main(stack=STACK):
    cases = sorted(d for d in glob.glob(os.path.join(stack, "*")) if os.path.isdir(d))
    print(f"Hydrodynamic power closure over [0, {TMAX}] s   "
          f"(100% = the readout's claimed work equals the fluid's actual gain)")
    print(f"{'case':>14} {'<P_BDIM> [W]':>13} {'E_BDIM [J]':>12} {'dE_k [J]':>11} "
          f"{'E_diss [J]':>11} {'received [J]':>12} {'CLOSURE':>9}")
    for c in cases:
        rd = _latest(c)
        if rd is None:
            continue
        r = closure(rd)
        print(f"{os.path.basename(c):>14} {r['P_bdim_mean']:13.4e} {r['E_bdim']:12.4e} "
              f"{r['dEk']:11.4e} {r['E_diss']:11.4e} {r['received']:12.4e} "
              f"{r['closure']*100:8.1f}%")
    print("\nA readout that over-reports drag claims more work than the water")
    print("received -> closure >> 100%.  The freq-cross study closed at 90-102%.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else STACK)
