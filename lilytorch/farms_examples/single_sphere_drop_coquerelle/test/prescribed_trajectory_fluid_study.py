#!/usr/bin/env python3
"""Fluid-only Heun/Euler study on a prescribed body trajectory.

The body trajectory (x,y,vx,vy vs time) is fixed from a reference run,
so all cases see identical kinematics. This isolates fluid-integrator
behavior from rigid-body coupling effects.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from adhoc_fsi_coquerelle import (
    SIM_CFG,
    OUT_ROOT,
    _build_solver_pars,
    _fluid_step_2d_like_bdimhandler,
    _load_base_bdim_yaml,
    _set_body_linearized_kinematics,
)
from lilytorch.src.solver import FluidSolver


REF_CASE = "heun_dt_0p00005"
STUDY_DIR = OUT_ROOT / "prescribed_traj_fluid"
T_END = 0.32

CASES = [
    {"name": "heun_dt_0p00005", "heun": True, "dt": 5.0e-5},
    {"name": "euler_dt_0p00005", "heun": False, "dt": 5.0e-5},
    {"name": "euler_dt_0p000025", "heun": False, "dt": 2.5e-5},
    {"name": "euler_dt_0p0000125", "heun": False, "dt": 1.25e-5},
]


def nt_from_dt(dt: float, t_end: float) -> int:
    return int(round(t_end / dt)) + 1


def load_reference_traj():
    p = OUT_ROOT / REF_CASE / "adhoc_timeseries.npz"
    if not p.exists():
        raise FileNotFoundError(f"Reference trajectory not found: {p}")
    a = np.load(p)
    return a["t"], a["x"], a["y"], a["vx"], a["vy"]


def run_case(case: dict, t_ref, x_ref, y_ref, vx_ref, vy_ref, rho_fluid, rho_body):
    dt = float(case["dt"])
    heun = bool(case["heun"])
    nt = nt_from_dt(dt, T_END)

    name = case["name"]
    case_dir = STUDY_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)

    base = _load_base_bdim_yaml(SIM_CFG)
    pars, dtype = _build_solver_pars(base, dt=dt, nt=nt, heun=heun)

    fs = FluidSolver(pars, dtype=dtype, custom_update=None, compute_forces=True)

    comp = fs.composite_body
    body = comp.bodies[0]

    t = np.arange(nt, dtype=np.float64) * dt
    x = np.interp(t, t_ref, x_ref)
    y = np.interp(t, t_ref, y_ref)
    vx = np.interp(t, t_ref, vx_ref)
    vy = np.interp(t, t_ref, vy_ref)

    fx = np.zeros(nt, dtype=np.float64)
    fy = np.zeros(nt, dtype=np.float64)
    fpx = np.zeros(nt, dtype=np.float64)
    fpy = np.zeros(nt, dtype=np.float64)
    fvx = np.zeros(nt, dtype=np.float64)
    fvy = np.zeros(nt, dtype=np.float64)

    for it in range(nt):
        _set_body_linearized_kinematics(
            body,
            state=type("S", (), {"x": float(x[it]), "y": float(y[it]), "vx": float(vx[it]), "vy": float(vy[it])})(),
            t0=float(t[it]),
        )

        comp.update(torch.tensor(float(t[it]), device=fs.device, dtype=fs.dtype), it, dt=dt)

        # fields required by forces_method2
        comp.sdf_vals = body.sdf_val.unsqueeze(0)
        comp.sdf_vals_u = body.sdf_u.unsqueeze(0)
        comp.sdf_vals_v = body.sdf_v.unsqueeze(0)
        comp.com_pos[0, 0] = body.com_pos[0]
        comp.com_pos[0, 1] = body.com_pos[1]

        fs._recompute_mu_normals()

        u, v, p = _fluid_step_2d_like_bdimhandler(
            fs,
            fs.u0, fs.v0, fs.p0,
            timestep=dt,
            rho_fluid=rho_fluid,
            rho_body=rho_body,
            use_heun=heun,
        )
        fs.u0, fs.v0, fs.p0 = u, v, p

        fs.forces_method2(fs.u0, fs.v0, fs.p0, it)

        fvx[it] = float(fs.friction_force_lin_x[0].item())
        fvy[it] = float(fs.friction_force_lin_y[0].item())
        fpx[it] = float(fs.pressure_force_x[0].item())
        fpy[it] = float(fs.pressure_force_y[0].item())
        fx[it] = fvx[it] + fpx[it]
        fy[it] = fvy[it] + fpy[it]

        if (it + 1) % max(1, nt // 10) == 0:
            print(f"[{name}] {it+1}/{nt} t={t[it]:.4f}s fy={fy[it]:.4f} p={fpy[it]:.4f} v={fvy[it]:.4f}")

    np.savez_compressed(
        case_dir / "fluid_forces.npz",
        t=t, x=x, y=y, vx=vx, vy=vy,
        fx=fx, fy=fy,
        fpx=fpx, fpy=fpy,
        fvx=fvx, fvy=fvy,
    )

    summary = {
        "case_name": name,
        "dt": dt,
        "nt": int(nt),
        "heun": heun,
        "fy_mean_y_1p2_0p8": float(np.mean(fy[(y <= 1.2) & (y >= 0.8)])),
        "fy_std_y_1p2_0p8": float(np.std(fy[(y <= 1.2) & (y >= 0.8)])),
        "fx_mean_y_1p2_0p8": float(np.mean(fx[(y <= 1.2) & (y >= 0.8)])),
    }
    with (case_dir / "fluid_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    t_ref, x_ref, y_ref, vx_ref, vy_ref = load_reference_traj()

    # Use same density parameters for all cases.
    base = _load_base_bdim_yaml(SIM_CFG)
    rho_fluid = float(base["solver"]["rho"])
    rho_body = float(base["solver"].get("rho_body", 800.0))

    summaries = []
    for case in CASES:
        summaries.append(run_case(case, t_ref, x_ref, y_ref, vx_ref, vy_ref, rho_fluid, rho_body))

    # Baseline comparison against Heun dt=5e-5 on identical trajectory.
    ref_file = STUDY_DIR / "heun_dt_0p00005" / "fluid_forces.npz"
    ref = np.load(ref_file)
    tref = ref["t"]
    fy_ref = ref["fy"]

    rows = []
    for case in CASES:
        f = np.load(STUDY_DIR / case["name"] / "fluid_forces.npz")
        fy_i = np.interp(tref, f["t"], f["fy"])
        y_i = np.interp(tref, f["t"], f["y"])
        m = (y_i <= 1.2) & (y_i >= 0.8)
        rms = float(np.sqrt(np.mean((fy_i[m] - fy_ref[m]) ** 2)))
        row = {
            "case_name": case["name"],
            "dt": case["dt"],
            "heun": case["heun"],
            "rms_fy_vs_heun_ref_window": rms,
        }
        rows.append(row)

    out_csv = STUDY_DIR / "prescribed_traj_force_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_name", "dt", "heun", "rms_fy_vs_heun_ref_window"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[saved] {out_csv}")
    print("\nRMS force error vs Heun ref (same trajectory, y in [1.2,0.8]):")
    for r in rows:
        print(f"- {r['case_name']}: {r['rms_fy_vs_heun_ref_window']:.6f}")


if __name__ == "__main__":
    main()
