"""Verify the energy balance  dE_k/dt = P_act - dissipation_rate.

Reads one freq-cross run and checks whether the energy budget closes.
If the residual is large, diagnoses where the missing energy goes.

Usage:
    python verify_energy_balance.py
    python verify_energy_balance.py --case slow_slow --tmax 0.3
"""

import argparse
import glob
import os

import h5py
import numpy as np
from farms_core.sensors.sensor_convention import sc

DEFAULT_STACK = "/data/andreaferrario/ns_data/zebrafish_freq_cross"
G = 9.81


def _latest_run_dir(case_dir: str):
    subs = sorted(d for d in glob.glob(os.path.join(case_dir, "*")) if os.path.isdir(d))
    for d in reversed(subs):
        if os.path.exists(os.path.join(d, "output", "simulation.hdf5")):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack", default=DEFAULT_STACK)
    ap.add_argument("--case", default="slow_slow")
    ap.add_argument("--tmax", type=float, default=0.3)
    ap.add_argument("--rho", type=float, default=1000.0)
    args = ap.parse_args()

    run_dir = _latest_run_dir(os.path.join(args.stack, args.case))
    if run_dir is None:
        raise SystemExit(f"No run found for {args.case}")

    # ── Load all data ──────────────────────────────────────────────────
    sim_path = os.path.join(run_dir, "output", "simulation.hdf5")
    diag_path = os.path.join(run_dir, "diagnostics.h5")

    with h5py.File(sim_path, "r") as f:
        # Link sensors — COM velocities and positions
        link = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        masses = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["masses"])
        # Joint sensors — torques and velocities
        j = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"])
        dt_sim = float(np.array(f["timestep"]))
        if "times" in f:
            sim_times = np.array(f["times"])[: link.shape[0]]
        else:
            sim_times = dt_sim * np.arange(link.shape[0])

    with h5py.File(diag_path, "r") as f:
        ke_raw = np.array(f["kinetic_energy"])          # per-unit-density [m^5/s^2]
        diss_raw = np.array(f["dissipation_rate"])      # per-unit-density [m^5/s^3]
        # Unmasked dissipation (whole grid) — only present on runs recorded after
        # the diagnostics.py update; used to gauge boundary-layer clipping.
        diss_unmasked_raw = (np.array(f["dissipation_rate_unmasked"])
                             if "dissipation_rate_unmasked" in f else None)

    # ── Align time axes ────────────────────────────────────────────────
    # Diagnostics are written every `diagnostics_every` solver steps.
    # The array is indexed by solver iteration; valid slots are finite.
    ke_valid = np.isfinite(ke_raw)
    ke_idx = np.where(ke_valid)[0]
    ke_t = ke_idx * dt_sim                               # diagnostic times [s]
    ke_J = args.rho * ke_raw[ke_valid]                    # kinetic energy [J]

    diss_valid = np.isfinite(diss_raw)
    diss_idx = np.where(diss_valid)[0]
    diss_t = diss_idx * dt_sim
    diss_W = args.rho * diss_raw[diss_valid]              # dissipation power [W]

    # ── Truncate to tmax ───────────────────────────────────────────────
    keep_ke = ke_t <= args.tmax
    ke_t, ke_J = ke_t[keep_ke], ke_J[keep_ke]
    keep_diss = diss_t <= args.tmax
    diss_t, diss_W = diss_t[keep_diss], diss_W[keep_diss]

    keep_sim = sim_times <= args.tmax
    sim_times = sim_times[keep_sim]

    # ── Mechanical actuator power ──────────────────────────────────────
    tau = j[: len(sim_times), :, sc.joint_torque]        # (T, n_joints)
    omega = j[: len(sim_times), :, sc.joint_velocity]
    P_act = np.abs(tau * omega).sum(axis=1)               # total [W]

    # ── dE_k/dt via central differences ────────────────────────────────
    dEk_dt = np.gradient(ke_J, ke_t, edge_order=2)

    # ── Interpolate dissipation onto KE time grid ──────────────────────
    diss_on_ke_t = np.interp(ke_t, diss_t, diss_W, left=0.0, right=diss_W[-1])

    # ── Energy balance residual ────────────────────────────────────────
    #   dE_k/dt = P_act - dissipation_rate + residual
    #   => residual = dE_k/dt - P_act + dissipation_rate
    P_act_on_ke_t = np.interp(ke_t, sim_times, P_act, left=0.0, right=P_act[-1])
    residual = dEk_dt - P_act_on_ke_t + diss_on_ke_t

    # ── Cumulative energy tracking ─────────────────────────────────────
    # E_mech = cumulative integral of P_act
    e_mech = np.concatenate([[0.0], np.cumsum(
        0.5 * (P_act[1:] + P_act[:-1]) * np.diff(sim_times)
    )])
    # E_diss = cumulative integral of dissipation_rate
    e_diss = np.concatenate([[0.0], np.cumsum(
        0.5 * (diss_W[1:] + diss_W[:-1]) * np.diff(diss_t)
    )])
    # E_k change from t=0
    delta_Ek = ke_J - ke_J[0]

    # ── Diagnostic prints ──────────────────────────────────────────────
    m_fish = masses.sum()
    print(f"Case: {args.case}")
    print(f"  Run dir: {run_dir}")
    print(f"  dt_sim = {dt_sim:.6f} s")
    print(f"  tmax = {args.tmax} s  ({len(sim_times)} sim steps, {len(ke_t)} diag samples)")
    print(f"  Fish mass = {m_fish:.6e} kg ({m_fish*1e3:.3f} g)")
    print()

    # Mean power values
    P_act_mean = float(np.mean(P_act))
    diss_mean = float(np.mean(diss_W))
    dEk_mean = float(np.mean(dEk_dt[1:]))  # skip t=0 edge artefact
    res_mean = float(np.mean(residual[1:]))
    res_rms = float(np.sqrt(np.mean(residual[1:]**2)))

    print(f"  <P_act>          = {P_act_mean:.4e} W")
    print(f"  <dissipation>    = {diss_mean:.4e} W")
    if diss_unmasked_raw is not None:
        du = args.rho * diss_unmasked_raw[diss_valid]
        du = du[keep_diss]
        print(f"  <diss unmasked>  = {float(np.mean(du)):.4e} W "
              f"(boundary-layer clip = {(np.mean(du)/diss_mean - 1)*100:.1f}%)")
    print(f"  <dE_k/dt>        = {dEk_mean:.4e} W")
    print(f"  ─────────────────────────────────")
    print(f"  <residual>       = {res_mean:.4e} W")
    print(f"  residual RMS     = {res_rms:.4e} W")
    print(f"  residual / P_act = {abs(res_mean)/P_act_mean*100:.1f}%")
    print()

    # Cumulative at tmax
    e_mech_final = float(e_mech[-1])
    e_diss_final = float(e_diss[-1])
    print(f"  At t = {args.tmax} s:")
    print(f"    E_mech  (∫P_act dt)   = {e_mech_final:.4e} J")
    print(f"    E_diss  (∫diss  dt)   = {e_diss_final:.4e} J")
    print(f"    ΔE_k    (E_k-E_k(0))  = {delta_Ek[-1]:.4e} J")
    print(f"    Balance: E_mech - E_diss - ΔE_k = {e_mech_final - e_diss_final - delta_Ek[-1]:.4e} J")
    print(f"    Unaccounted fraction = {(e_mech_final - e_diss_final - delta_Ek[-1])/e_mech_final*100:.1f}%")
    print()

    # COT comparison
    # Need COM distance
    pos = link[: len(sim_times), :, sc.link_com_position_x : sc.link_com_position_z + 1]
    com_xy = (masses[None, :, None] * pos[:, :, :2]).sum(axis=1) / m_fish
    d_xy = np.sqrt((com_xy[:, 0] - com_xy[0, 0])**2 + (com_xy[:, 1] - com_xy[0, 1])**2)
    far = d_xy > 1e-4
    if far.any():
        cot_mech = e_mech / (m_fish * G * d_xy)
        cot_diss = np.interp(sim_times, diss_t, e_diss, left=0.0, right=e_diss[-1])
        cot_diss = cot_diss / (m_fish * G * d_xy)
        half = len(sim_times) // 2
        print(f"  COT_mech (ss)  = {np.nanmean(cot_mech[half:][far[half:]]):.4f}")
        print(f"  COT_diss (ss)  = {np.nanmean(cot_diss[half:][far[half:]]):.4f}")
        print(f"  COT ratio      = {np.nanmean(cot_mech[half:][far[half:]])/np.nanmean(cot_diss[half:][far[half:]]):.1f}x")

    # ── Plot ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: Instantaneous power balance
    ax = axes[0]
    ax.plot(ke_t, P_act_on_ke_t, label=r"$P_{\mathrm{act}}$ (joints)", color="C0")
    ax.plot(ke_t, diss_on_ke_t, label=r"$\dot{E}_{\mathrm{diss}}$ (CFD, masked)", color="C3")
    ax.plot(ke_t, dEk_dt, label=r"$dE_k/dt$", color="C2", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Power [W]")
    ax.set_title(f"Instantaneous power balance — {args.case}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: Residual
    ax = axes[1]
    ax.plot(ke_t, residual, color="C4", label="residual = dE_k/dt − P_act + dissipation")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(res_mean, color="C4", ls="--", lw=0.8, label=f"mean = {res_mean:.2e} W")
    ax.set_ylabel("Residual [W]")
    ax.set_title("Energy balance residual  (should ≈ 0)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 3: Cumulative energy
    ax = axes[2]
    ax.plot(sim_times, e_mech, label=r"$E_{\mathrm{mech}}$ (∫P_act dt)", color="C0")
    ax.plot(diss_t, e_diss, label=r"$E_{\mathrm{diss}}$ (∫diss dt)", color="C3")
    ax.plot(ke_t, delta_Ek, label=r"$\Delta E_k$", color="C2")
    ax.plot(ke_t, e_mech_final - delta_Ek[-1] - e_diss_final + delta_Ek * 0,
            color="gray", lw=0.5, alpha=0)  # dummy
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("Cumulative energy")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    # Add text box with unaccounted fraction
    unaccounted = (e_mech_final - e_diss_final - delta_Ek[-1]) / e_mech_final * 100
    ax.text(0.98, 0.05, f"Unaccounted: {unaccounted:.1f}% of E_mech",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    out_png = os.path.join(args.stack, f"energy_balance_{args.case}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot: {out_png}")


if __name__ == "__main__":
    main()
