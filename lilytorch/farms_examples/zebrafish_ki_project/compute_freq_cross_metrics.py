"""Compute swim speed + total fluid kinetic energy for the freq-cross runs.

Reads each case produced by ``run_freq_cross.py`` and reports:

  * forward COM swim speed (PCA body axis), from ``output/simulation.hdf5``
    -- mean over the whole run and over the last half (quasi-steady),
       in m/s and body-lengths/s.
  * total fluid kinetic energy E_k(t) = rho * 0.5 * h^d * sum|u|^2,
    from ``diagnostics.h5`` -- mean / peak / final, and the time-integral
    int E_k dt (a proxy for cumulative energy in the wake).
  * cost of transport  COT(t) = E_mech(t) / (m * g * d(t)) -- dimensionless,
    where E_mech(t) = int Sum_j |tau_j * omega_j| dt is the cumulative
    mechanical actuator work, m the total fish mass, and d(t) the x-y COM
    swim distance.  Both numerator and denominator are cumulative, so COT is
    stationary in steady state and fairly ranks the four gaits.

Outputs a summary table to stdout, a CSV, and a 3-panel comparison plot
(speed + fluid KE + COT vs time) under the stack root.

Usage
-----
    python compute_freq_cross_metrics.py
    python compute_freq_cross_metrics.py --stack /data/andreaferrario/ns_data/zebrafish_freq_cross
    python compute_freq_cross_metrics.py --bl 0.017
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import h5py
import numpy as np
import yaml
from farms_core.sensors.sensor_convention import sc

from lilytorch.util.metrics import compute_speed_PCA

DEFAULT_STACK = "/data/andreaferrario/ns_data/zebrafish_freq_cross"
CASE_ORDER = ["slow_slow", "fast_fast", "slow_fast", "fast_slow"]
G = 9.81  # m/s²


def _latest_run_dir(case_dir: str):
    """Newest timestamped sub-directory holding output/simulation.hdf5."""
    subs = sorted(
        d for d in glob.glob(os.path.join(case_dir, "*")) if os.path.isdir(d)
    )
    for d in reversed(subs):
        if os.path.exists(os.path.join(d, "output", "simulation.hdf5")):
            return d
    return None


def _read_rho(run_dir: str, default: float = 1000.0) -> float:
    """Fluid density from the solver parameters.yaml (falls back to water)."""
    path = os.path.join(run_dir, "parameters.yaml")
    if os.path.exists(path):
        try:
            with open(path) as f:
                pars = yaml.unsafe_load(f)
            return float(pars["solver"]["rho"])
        except Exception:
            pass
    return default


def _speed_and_com(run_dir: str):
    """Return (times, v_forward, v_lateral, com_xy) from the FARMS link sensors.

    com_xy is shape (T, 2) — mass-weighted x-y COM position at each timestep.
    """
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        link   = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        masses = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["masses"])
        if "times" in f:
            times = np.array(f["times"])[: link.shape[0]]
        else:
            dt = float(np.array(f["timestep"]))
            times = dt * np.arange(link.shape[0])
    vel = link[:, :, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]
    pos = link[:, :, sc.link_com_position_x : sc.link_com_position_z + 1]
    v_fwd, v_lat = compute_speed_PCA(pos, vel)
    # mass-weighted COM in x-y
    m_total = masses.sum()
    com_xy = (masses[None, :, None] * pos[:, :, :2]).sum(axis=1) / m_total
    return times, np.asarray(v_fwd), np.asarray(v_lat), com_xy, float(m_total)


def _fluid_ke(run_dir: str, rho: float):
    """Return (idx, E_k[J]) from diagnostics.h5 (NaN slots dropped).

    idx is the solver iteration index (aligns with simulation timestep array).
    """
    path = os.path.join(run_dir, "diagnostics.h5")
    if not os.path.exists(path):
        return None, None
    with h5py.File(path, "r") as f:
        ke = np.array(f["kinetic_energy"])
    valid = np.isfinite(ke)
    if not valid.any():
        return None, None
    idx = np.where(valid)[0]
    return idx, rho * ke[idx]


def _joint_power(run_dir: str):
    """Return (times, P[W]) — total mechanical actuator power Σ_j |τ_j·ω_j|.

    Uses the realized joint torque (sc.joint_torque) and joint angular velocity
    (sc.joint_velocity) from the FARMS joint sensors.  The absolute value is
    taken per joint (no elastic energy recovery — standard cost-of-transport
    assumption), then summed over joints.
    """
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        j = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"])
        if "times" in f:
            times = np.array(f["times"])[: j.shape[0]]
        else:
            dt = float(np.array(f["timestep"]))
            times = dt * np.arange(j.shape[0])
    tau   = j[:, :, sc.joint_torque]      # (T, n_joints)
    omega = j[:, :, sc.joint_velocity]
    power = np.abs(tau * omega).sum(axis=1)   # (T,) total mechanical power [W]
    return times, power


def _dissipation_energy(run_dir: str, rho: float, dt: float):
    """Return (t_diag, E_diss[J]) — cumulative dissipated energy from diagnostics.h5.

    Reads the per-unit-density dissipation rate ν·hᵈ·Σ(μ₀|S̄|²) [m⁵/s³],
    multiplies by *rho* for Watts, then cumulatively integrates for Joules.
    The time axis is *idx* · *dt* where *idx* is the solver iteration index
    (written every ``diagnostics_every`` steps).
    """
    path = os.path.join(run_dir, "diagnostics.h5")
    if not os.path.exists(path):
        return None, None
    with h5py.File(path, "r") as f:
        diss_rate = np.array(f["dissipation_rate"])
    valid = np.isfinite(diss_rate)
    if not valid.any():
        return None, None
    idx = np.where(valid)[0]
    power_w = rho * diss_rate[valid]                      # (N,) Watts
    t_diag = idx * dt                                      # (N,) seconds
    e_diss = _cumtrapz0(power_w, t_diag)                   # (N,) Joules
    return t_diag, e_diss


def _bdim_power(run_dir: str):
    """Return (times, P_bdim[W]) — hydrodynamic power the body delivers to the fluid.

    Reads the per-link hydrodynamic forces/torques from ``drags.h5`` (force and
    torque *on the body* from the fluid, pressure + viscous, torque about each
    link COM) and the link COM linear/angular velocities from the FARMS link
    sensors.  The power injected into the fluid is the *reaction* of the work the
    fluid does on the body,

        P_bdim(t) = − Σ_links [ F_i·v_i + τ_i·ω_i ]                         [W]

    where the sum runs over translational (F·v) and rotational (τ·ω) channels.
    This is an independent estimate of the fluid power that, in the wall-free
    window, closes the energy balance with dissipation + dE_k/dt (verified on
    slow_slow to ~2 %).  Returns ``(None, None)`` if ``drags.h5`` is absent.
    """
    path = os.path.join(run_dir, "drags.h5")
    if not os.path.exists(path):
        return None, None
    with h5py.File(path, "r") as f:
        # (n_links, 3, nt) — force/torque ON the body from the fluid.
        F = np.array(f["pressure_drags"]) + np.array(f["viscous_drags"])
        T = np.array(f["pressure_torques"]) + np.array(f["viscous_torques"])
    F = np.transpose(F, (2, 0, 1))           # (nt, n_links, 3)
    T = np.transpose(T, (2, 0, 1))
    with h5py.File(os.path.join(run_dir, "output", "simulation.hdf5"), "r") as f:
        link = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        if "times" in f:
            times = np.array(f["times"])[: link.shape[0]]
        else:
            dt = float(np.array(f["timestep"]))
            times = dt * np.arange(link.shape[0])
    v = link[:, :, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]
    w = link[:, :, sc.link_com_velocity_ang_x : sc.link_com_velocity_ang_z + 1]
    n = min(F.shape[0], v.shape[0])
    F, T, v, w, times = F[:n], T[:n], v[:n], w[:n], times[:n]
    p_bdim = -((F * v).sum(axis=(1, 2)) + (T * w).sum(axis=(1, 2)))   # (n,) [W]
    return times, p_bdim


def _cumtrapz0(y, x):
    """Cumulative trapezoidal integral of y over x, prepended with 0 (len == len(y))."""
    if len(y) < 2:
        return np.zeros_like(y, dtype=float)
    incr = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(incr)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack", default=DEFAULT_STACK, help="freq-cross stack root.")
    ap.add_argument("--bl", type=float, default=0.017,
                    help="Body length [m] for BL/s normalisation (default 0.017).")
    ap.add_argument("--out", default=None, help="Output prefix (default: <stack>/freq_cross).")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                    help="x-axis limits [s] for all panels (e.g. --xlim 0 2).")
    ap.add_argument("--ylim-speed", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                    help="y-axis limits [BL/s] for the speed panel.")
    ap.add_argument("--ylim-ke", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                    help="y-axis limits [J] for the fluid KE panel.")
    ap.add_argument("--ylim-cot", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                    help="y-axis limits for the COT panel (dimensionless).")
    ap.add_argument("--cot-logy", action="store_true",
                    help="Use a log y-axis on the COT panel.")
    ap.add_argument("--min-dist", type=float, default=1e-4,
                    help="Minimum swim distance [m] before COT is computed (avoids t=0 zero).")
    ap.add_argument("--tmax", type=float, default=None,
                    help="Truncate ALL data (speed, KE, COT, distance) to t <= tmax [s] "
                         "before computing metrics. Use to exclude the wall-contact window "
                         "(e.g. --tmax 0.3).")
    args = ap.parse_args()

    out_prefix = args.out or os.path.join(args.stack, "freq_cross")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_v, ax_e, ax_c) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    rows = []

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ci, case in enumerate(CASE_ORDER):
        color = color_cycle[ci % len(color_cycle)]
        case_dir = os.path.join(args.stack, case)
        if not os.path.isdir(case_dir):
            print(f"[skip] no directory for case {case}: {case_dir}")
            continue
        run_dir = _latest_run_dir(case_dir)
        if run_dir is None:
            print(f"[skip] no completed run under {case_dir}")
            continue

        rho = _read_rho(run_dir)
        times, v_fwd, v_lat, com_xy, m_fish = _speed_and_com(run_dir)
        dt = float(times[1] - times[0]) if len(times) > 1 else 5e-4

        # Truncate to the wall-free window (t <= tmax) before any metric.
        if args.tmax is not None:
            keep = times <= args.tmax
            times, v_fwd, v_lat = times[keep], v_fwd[keep], v_lat[keep]
            com_xy = com_xy[keep]

        half = len(v_fwd) // 2
        v_mean    = float(np.mean(v_fwd))
        v_mean_ss = float(np.mean(v_fwd[half:]))

        # ── Fluid kinetic energy (context panel) ──────────────────────
        ke_idx, ke = _fluid_ke(run_dir, rho)
        if ke is not None:
            ke_t = ke_idx * dt
            if args.tmax is not None:
                keep = ke_t <= args.tmax
                ke_t, ke = ke_t[keep], ke[keep]
            ke_mean  = float(np.mean(ke))
            ke_peak  = float(np.max(ke))
            ke_final = float(ke[-1])
            ax_e.plot(ke_t, ke, color=color, label=case)
        else:
            ke_mean = ke_peak = ke_final = float("nan")
            print(f"[warn] no diagnostics.h5 for {case} (fluid KE unavailable)")

        # ── Cost of transport  COT(t) = E_mech(t) / (m·g·d(t)) ────────
        #    E_mech : cumulative actuator work  ∫ Σ|τ·ω| dt   [J]
        #    d      : x-y COM displacement from start          [m]
        #    Both cumulative → COT is stationary in steady state, so its
        #    quasi-steady value fairly ranks the four gaits.
        tj, power = _joint_power(run_dir)
        n = min(len(tj), len(com_xy))
        tj, power, com = tj[:n], power[:n], com_xy[:n]
        e_mech = _cumtrapz0(power, tj)                       # (n,) [J]
        d_xy = np.sqrt((com[:, 0] - com[0, 0]) ** 2 +
                       (com[:, 1] - com[0, 1]) ** 2)         # (n,) [m]
        far = d_xy > args.min_dist
        cot = np.full(n, np.nan)
        cot[far] = e_mech[far] / (m_fish * G * d_xy[far])
        ax_c.plot(tj, cot, color=color, label=case)

        e_mech_final = float(e_mech[-1])
        d_final = float(d_xy[-1])
        # Quasi-steady COT: mean over the last half of the (valid) trajectory.
        half_far = far.copy()
        half_far[: n // 2] = False
        cot_ss = float(np.nanmean(cot[half_far])) if half_far.any() else float("nan")
        cot_final = float(cot[-1]) if far[-1] else float("nan")

        # ── Dissipation CoT  COT_diss(t) = E_diss(t) / (m·g·d(t)) ──────
        #    E_diss  : cumulative dissipated energy  ρ·∫ν·hᵈ·Σ(μ₀|S̄|²) dt [J]
        #    At steady state, COT_diss ≈ COT_mech by the energy balance
        #    dE_k/dt = P_act − dissipation_rate; with 0 cycle-averaged dE_k/dt.
        diss_t, e_diss = _dissipation_energy(run_dir, rho, dt)
        if diss_t is not None and e_diss is not None:
            # Truncate to tmax (same window as all other data).
            if args.tmax is not None:
                keep_d = diss_t <= args.tmax
                diss_t, e_diss = diss_t[keep_d], e_diss[keep_d]
            # Interpolate E_diss onto the simulation time grid for CoT.
            e_diss_sim = np.interp(tj, diss_t, e_diss, left=0.0, right=e_diss[-1])
            cot_diss = np.full(n, np.nan)
            cot_diss[far] = e_diss_sim[far] / (m_fish * G * d_xy[far])
            # Dashed line for dissipation CoT on the same panel.
            ax_c.plot(tj, cot_diss, color=color, linestyle="--", alpha=0.7)
            cot_diss_ss = float(np.nanmean(cot_diss[half_far])) if half_far.any() else float("nan")
            cot_diss_final = float(cot_diss[-1]) if far[-1] else float("nan")
            e_diss_final = float(e_diss[-1]) if len(e_diss) > 0 else float("nan")
        else:
            cot_diss_ss = cot_diss_final = e_diss_final = float("nan")

        # ── BDIM CoT  COT_bdim(t) = E_bdim(t) / (m·g·d(t)) ─────────────
        #    P_bdim  : hydrodynamic power into the fluid  −Σ(F·v + τ·ω) [W]
        #    E_bdim  : cumulative ∫ P_bdim dt                            [J]
        #    Independent of the CFD dissipation integral; in steady state
        #    COT_bdim ≈ COT_diss (both measure fluid power), so the dotted
        #    line cross-checks the dashed one.
        tb, p_bdim = _bdim_power(run_dir)
        if tb is not None and p_bdim is not None:
            if args.tmax is not None:
                keep_b = tb <= args.tmax
                tb, p_bdim = tb[keep_b], p_bdim[keep_b]
            e_bdim = _cumtrapz0(p_bdim, tb)                  # (.,) [J]
            e_bdim_sim = np.interp(tj, tb, e_bdim, left=0.0, right=e_bdim[-1])
            cot_bdim = np.full(n, np.nan)
            cot_bdim[far] = e_bdim_sim[far] / (m_fish * G * d_xy[far])
            # Dotted line for BDIM CoT on the same panel.
            ax_c.plot(tj, cot_bdim, color=color, linestyle=":", alpha=0.9)
            cot_bdim_ss = float(np.nanmean(cot_bdim[half_far])) if half_far.any() else float("nan")
            cot_bdim_final = float(cot_bdim[-1]) if far[-1] else float("nan")
            e_bdim_final = float(e_bdim[-1]) if len(e_bdim) > 0 else float("nan")
            p_bdim_mean = float(np.mean(p_bdim))
        else:
            cot_bdim_ss = cot_bdim_final = e_bdim_final = p_bdim_mean = float("nan")
            print(f"[warn] no drags.h5 for {case} (BDIM CoT unavailable)")

        ax_v.plot(times, v_fwd / args.bl, color=color, label=case)

        rows.append({
            "case": case,
            "run_dir": run_dir,
            "rho": rho,
            "fish_mass_kg": m_fish,
            "v_fwd_mean_mps": v_mean,
            "v_fwd_mean_ss_mps": v_mean_ss,
            "v_fwd_mean_ss_BLps": v_mean_ss / args.bl,
            "fluid_KE_mean_J": ke_mean,
            "fluid_KE_peak_J": ke_peak,
            "fluid_KE_final_J": ke_final,
            "E_mech_final_J": e_mech_final,
            "dist_final_m": d_final,
            "COT_ss": cot_ss,
            "COT_final": cot_final,
            "E_diss_final_J": e_diss_final,
            "COT_diss_ss": cot_diss_ss,
            "COT_diss_final": cot_diss_final,
            "P_bdim_mean_W": p_bdim_mean,
            "E_bdim_final_J": e_bdim_final,
            "COT_bdim_ss": cot_bdim_ss,
            "COT_bdim_final": cot_bdim_final,
        })

    ax_v.set_ylabel(r"$V_{\mathrm{fwd}}$ [BL/s]")
    ax_v.set_title(f"Forward swim speed  (BL = {args.bl*1e3:.1f} mm)")
    ax_v.legend(); ax_v.grid(alpha=0.3)

    ax_e.set_ylabel(r"Fluid $E_k$ [J]")
    ax_e.set_title("Total fluid kinetic energy")
    ax_e.legend(); ax_e.grid(alpha=0.3)

    ax_c.set_ylabel(r"COT $= E/(m\,g\,d)$  [–]")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_title("Cost of transport  (solid: mech  |  dashed: dissipation  |  dotted: BDIM)")
    if args.cot_logy:
        ax_c.set_yscale("log")
    ax_c.legend(); ax_c.grid(alpha=0.3)

    xlim = args.xlim if args.xlim is not None else (
        (0.0, args.tmax) if args.tmax is not None else None)
    if xlim is not None:
        for ax in (ax_v, ax_e, ax_c):
            ax.set_xlim(xlim)
    if args.ylim_speed is not None:
        ax_v.set_ylim(args.ylim_speed)
    if args.ylim_ke is not None:
        ax_e.set_ylim(args.ylim_ke)
    if args.ylim_cot is not None:
        ax_c.set_ylim(args.ylim_cot)

    fig.suptitle("Zebrafish frequency-cross: speed, fluid KE & cost of transport", y=1.0)
    fig.tight_layout()
    png = out_prefix + "_metrics.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot: {png}")

    if rows:
        csv_path = out_prefix + "_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved table: {csv_path}\n")

        # Pretty stdout table
        hdr = ["case", "v_fwd_ss[BL/s]", "v_fwd_ss[m/s]",
               "E_mech_f[J]", "E_diss_f[J]", "dist_f[m]",
               "COT_ss[-]", "COT_diss_ss[-]", "COT_bdim_ss[-]"]
        print("  ".join(f"{h:>14}" for h in hdr))
        for r in rows:
            print("  ".join(f"{v:>14}" for v in [
                r["case"],
                f"{r['v_fwd_mean_ss_BLps']:.4f}",
                f"{r['v_fwd_mean_ss_mps']:.5f}",
                f"{r['E_mech_final_J']:.4e}",
                f"{r['E_diss_final_J']:.4e}",
                f"{r['dist_final_m']:.4e}",
                f"{r['COT_ss']:.4e}",
                f"{r['COT_diss_ss']:.4e}",
                f"{r['COT_bdim_ss']:.4e}",
            ]))
    else:
        print("No runs found — launch run_freq_cross.py first.")


if __name__ == "__main__":
    main()
