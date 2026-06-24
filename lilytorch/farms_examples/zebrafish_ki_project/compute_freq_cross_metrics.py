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

        half = len(v_fwd) // 2
        v_mean    = float(np.mean(v_fwd))
        v_mean_ss = float(np.mean(v_fwd[half:]))

        # ── Fluid kinetic energy (context panel) ──────────────────────
        ke_idx, ke = _fluid_ke(run_dir, rho)
        if ke is not None:
            ke_t = ke_idx * dt
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
        })

    ax_v.set_ylabel(r"$V_{\mathrm{fwd}}$ [BL/s]")
    ax_v.set_title(f"Forward swim speed  (BL = {args.bl*1e3:.1f} mm)")
    ax_v.legend(); ax_v.grid(alpha=0.3)

    ax_e.set_ylabel(r"Fluid $E_k$ [J]")
    ax_e.set_title("Total fluid kinetic energy")
    ax_e.legend(); ax_e.grid(alpha=0.3)

    ax_c.set_ylabel(r"COT $= E_{\mathrm{mech}}/(m\,g\,d)$  [–]")
    ax_c.set_xlabel("Time [s]")
    ax_c.set_title("Cost of transport  (cumulative actuator work / weight·distance)")
    if args.cot_logy:
        ax_c.set_yscale("log")
    ax_c.legend(); ax_c.grid(alpha=0.3)

    if args.xlim is not None:
        for ax in (ax_v, ax_e, ax_c):
            ax.set_xlim(args.xlim)
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
               "E_mech_f[J]", "dist_f[m]", "COT_ss[-]", "COT_final[-]"]
        print("  ".join(f"{h:>14}" for h in hdr))
        for r in rows:
            print("  ".join(f"{v:>14}" for v in [
                r["case"],
                f"{r['v_fwd_mean_ss_BLps']:.4f}",
                f"{r['v_fwd_mean_ss_mps']:.5f}",
                f"{r['E_mech_final_J']:.4e}",
                f"{r['dist_final_m']:.4e}",
                f"{r['COT_ss']:.4e}",
                f"{r['COT_final']:.4e}",
            ]))
    else:
        print("No runs found — launch run_freq_cross.py first.")


if __name__ == "__main__":
    main()
