#!/usr/bin/env python3
"""Standalone 2-D Coquerelle FSI (no MuJoCo) for Heun/Euler investigation.

This script runs a single rigid circle coupled to the lilytorch 2-D BDIM
fluid solver using the same fluid-step logic as BDIMhandler._fluid_step_2d,
but updates body motion with a direct Newton ODE integrator.

Goal: isolate fluid-integrator effects from MuJoCo two-way coupling.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

from lilytorch.src.solver import FluidSolver


ROOT = Path(__file__).resolve().parent
SIM_CFG = ROOT / "simulation_config.yaml"
SPHERE_SDF = ROOT / "sphere.sdf"
OUT_ROOT = Path("/data/andreaferrario/ns_data/coquerelle_adhoc_study")


@dataclass
class RigidState:
    x: float
    y: float
    vx: float
    vy: float


DEFAULT_CASES = [
    {"name": "heun_dt_0p00005", "heun": True, "dt": 5.0e-5},
    {"name": "euler_dt_0p00005", "heun": False, "dt": 5.0e-5},
    {"name": "euler_dt_0p000025", "heun": False, "dt": 2.5e-5},
    {"name": "euler_dt_0p0000125", "heun": False, "dt": 1.25e-5},
]


def _case_nt(dt: float, t_end: float) -> int:
    return int(round(t_end / dt)) + 1


def _load_base_bdim_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ext in cfg.get("extensions", []):
        if ext.get("loader") == "lilytorch.integration.extensions.FluidExtension":
            return copy.deepcopy(ext["config"]["bdim_yaml"])
    raise KeyError("FluidExtension bdim_yaml not found in simulation_config.yaml")


def _build_solver_pars(base: dict, dt: float, nt: int, heun: bool) -> tuple[dict, torch.dtype]:
    pars = copy.deepcopy(base)

    solver = pars["solver"]
    solver["dt"] = float(dt)
    solver["nt"] = int(nt)
    solver["heun"] = bool(heun)

    # Use same method labels as FluidSolver standalone path.
    solver["time_integration"] = "heun" if heun else "euler"

    # Keep solver I/O off in this ad-hoc script (we save custom outputs).
    out = pars.setdefault("output", {})
    out["save_frames"] = False
    out["save_every"] = max(1, int(nt))
    out["save"] = False
    out["save_uv"] = False
    out.setdefault("vmin", -50)
    out.setdefault("vmax", 50)
    out.setdefault("save_path", "")

    # Replace FARMS multi-animat body with one analytical circle.
    # Circle is centred at (x, y) = (1, 4) initially; kinematics are
    # updated each time-step by replacing these functions with local
    # linear predictors around current state.
    pars["body"] = {
        "type": "composite_analytical",
        "plotting": False,
        "sdf": ["lambda x, y: circle(x, y, xt=0, yt=0, r=0.125)"],
        "update_maps": [
            {
                "rotation": "lambda t: 0.0",
                "translation": ["lambda t: 1.0", "lambda t: 4.0"],
            }
        ],
    }

    dtype_name = str(solver.get("dtype", "float64")).lower()
    dtype = torch.float64 if dtype_name == "float64" else torch.float32

    return pars, dtype


def _sphere_mass_radius_inertia(path: Path) -> tuple[float, float, float]:
    root = ET.parse(path).getroot()

    mass = float(root.find(".//inertial/mass").text)
    radius = float(root.find(".//collision/geometry/sphere/radius").text)

    # 2-D xz plane out-of-plane axis inertia in SDF is symmetric; use iyy.
    iyy = float(root.find(".//inertial/inertia/iyy").text)
    return mass, radius, iyy


def _compute_variable_density_coefficients(fs: FluidSolver, timestep: float,
                                           rho_fluid: float, rho_body: float):
    drho = rho_fluid - rho_body
    ch = timestep * fs.mu0_all_u / (rho_body + drho * fs.mu0_all_u)
    cv = timestep * fs.mu0_all_v / (rho_body + drho * fs.mu0_all_v)
    rho_cc = rho_body + drho * fs.mu0_all
    ch_cc = timestep / rho_cc
    return ch, cv, ch_cc


def _fluid_step_2d_like_bdimhandler(fs: FluidSolver, u, v, p, timestep: float,
                                    rho_fluid: float, rho_body: float, use_heun: bool):
    _bdim = fs._bdim_meta_compiled
    _h = fs.h
    comp = fs.composite_body

    ch, cv, ch_cc = _compute_variable_density_coefficients(fs, timestep, rho_fluid, rho_body)

    def _advect_bdim(u_in, v_in, nu_t=None, u_rebase=None, v_rebase=None):
        up, vp = fs.adv_diff_solver.solve(u_in, v_in, nu_t=nu_t)
        up = up.clone()
        vp = vp.clone()

        if u_rebase is not None:
            up = u_rebase + (up - u_in)
        if v_rebase is not None:
            vp = v_rebase + (vp - v_in)

        up = _bdim(
            up, fs.mu0_all_u,
            comp.body_u, fs.mu1_all_u,
            fs.normal_x_u, fs.normal_y_u, _h, 2,
        ).clone()
        vp = _bdim(
            vp, fs.mu0_all_v,
            comp.body_v, fs.mu1_all_v,
            fs.normal_x_v, fs.normal_y_v, _h, 2,
        ).clone()

        fs.adv_diff_solver.set_BCs(up, vp)
        return up, vp

    nu_t = fs._compute_nu_t(u, v)

    if use_heun:
        # Predictor
        up1, vp1 = _advect_bdim(u, v, nu_t=nu_t)
        u1, v1, p1 = fs.project(up1, vp1, p, ch=ch, cv=cv, ch_cc=ch_cc)
        fs.adv_diff_solver.set_BCs(u1, v1)

        # Corrector
        nu_t2 = fs._compute_nu_t(u1, v1)
        up2, vp2 = _advect_bdim(u1, v1, nu_t=nu_t2, u_rebase=u, v_rebase=v)

        u_avg = 0.5 * (u1 + up2)
        v_avg = 0.5 * (v1 + vp2)
        fs.adv_diff_solver.set_BCs(u_avg, v_avg)

        u_out, v_out, p_out = fs.project(
            u_avg, v_avg, p1,
            ch=0.5 * ch,
            cv=0.5 * cv,
            ch_cc=0.5 * ch_cc,
        )
    else:
        up, vp = _advect_bdim(u, v, nu_t=nu_t)
        u_out, v_out, p_out = fs.project(up, vp, p, ch=ch, cv=cv, ch_cc=ch_cc)

    if fs.use_sponge:
        u_out, v_out = fs.apply_sponge_damping(u_out, v_out)
    if fs.use_yield_damping:
        u_out, v_out = fs.apply_yield_damping(u_out, v_out)

    fs.adv_diff_solver.set_BCs(u_out, v_out)
    return u_out, v_out, p_out


def _set_body_linearized_kinematics(body, state: RigidState, t0: float):
    # Local linear model around current step; gives body.update() both
    # position and consistent autograd velocity derivatives.
    body.update_theta = (lambda t, t0=t0: 0.0)
    body.update_translation = (
        lambda t, x=state.x, vx=state.vx, t0=t0: x + vx * (t - t0),
        lambda t, y=state.y, vy=state.vy, t0=t0: y + vy * (t - t0),
    )


def run_case(case_name: str, dt: float, t_end: float, heun: bool) -> dict:
    nt = _case_nt(dt, t_end)
    case_dir = OUT_ROOT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    base = _load_base_bdim_yaml(SIM_CFG)
    pars, dtype = _build_solver_pars(base, dt=dt, nt=nt, heun=heun)

    fs = FluidSolver(pars, dtype=dtype, custom_update=None, compute_forces=True)

    # Physical body properties from SDF.
    mass, radius, inertia = _sphere_mass_radius_inertia(SPHERE_SDF)

    # Same gravity and buoyancy logic used in BDIMhandler (2-D xz mode).
    g = float(pars.get("solver", {}).get("g", -980.0))
    if "gravity" in yaml.safe_load(SIM_CFG.read_text()).get("physics", {}):
        g = float(yaml.safe_load(SIM_CFG.read_text())["physics"]["gravity"][2])

    rho_fluid = float(pars["solver"]["rho"])
    rho_body = float(pars["solver"].get("rho_body", 800.0))
    water_surface = float(pars["solver"].get("ymax", 0.0))

    state = RigidState(x=1.0, y=4.0, vx=0.0, vy=0.0)

    comp = fs.composite_body
    body = comp.bodies[0]

    times = np.zeros(nt, dtype=np.float64)
    x_hist = np.zeros(nt, dtype=np.float64)
    y_hist = np.zeros(nt, dtype=np.float64)
    vx_hist = np.zeros(nt, dtype=np.float64)
    vy_hist = np.zeros(nt, dtype=np.float64)
    fx_hist = np.zeros(nt, dtype=np.float64)
    fy_hist = np.zeros(nt, dtype=np.float64)
    buoy_hist = np.zeros(nt, dtype=np.float64)

    for it in range(nt):
        t = it * dt
        times[it] = t

        # 1) Update body kinematics fields used by BDIM.
        _set_body_linearized_kinematics(body, state, t0=t)
        comp.update(torch.tensor(t, device=fs.device, dtype=fs.dtype), it, dt=dt)

        # forces_method2 expects stacked per-body SDFs and COM positions
        # (the multi-animat path fills these in BDIMhandler._update_2d).
        comp.sdf_vals = body.sdf_val.unsqueeze(0)
        comp.sdf_vals_u = body.sdf_u.unsqueeze(0)
        comp.sdf_vals_v = body.sdf_v.unsqueeze(0)
        comp.com_pos[0, 0] = body.com_pos[0]
        comp.com_pos[0, 1] = body.com_pos[1]

        # 2) Recompute mu and normals from current body position.
        fs._recompute_mu_normals()

        # 3) Fluid step (Heun or Euler) with variable-density coefficients.
        u, v, p = _fluid_step_2d_like_bdimhandler(
            fs,
            fs.u0, fs.v0, fs.p0,
            timestep=dt,
            rho_fluid=rho_fluid,
            rho_body=rho_body,
            use_heun=heun,
        )
        fs.u0, fs.v0, fs.p0 = u, v, p

        # 4) Fluid forces on body.
        fs.forces_method2(fs.u0, fs.v0, fs.p0, it)
        fx = float((fs.friction_force_lin_x[0] + fs.pressure_force_x[0]).item())
        fy = float((fs.friction_force_lin_y[0] + fs.pressure_force_y[0]).item())

        # FARMS-style buoyancy (same formula as BDIMhandler _apply_forces_2d).
        buoy = 0.0
        if state.y - radius < water_surface:
            frac = min((water_surface + radius - state.y) / (2.0 * radius), 1.0)
            buoy = -rho_fluid * mass * g / rho_body * frac

        # 5) Rigid-body ODE update (symplectic Euler).
        ax = fx / mass
        ay = (fy + buoy + mass * g) / mass

        state.vx += dt * ax
        state.vy += dt * ay
        state.x += dt * state.vx
        state.y += dt * state.vy

        x_hist[it] = state.x
        y_hist[it] = state.y
        vx_hist[it] = state.vx
        vy_hist[it] = state.vy
        fx_hist[it] = fx
        fy_hist[it] = fy
        buoy_hist[it] = buoy

        if (it + 1) % max(1, nt // 10) == 0:
            print(
                f"[{case_name}] {it+1}/{nt} t={t:.4f}s "
                f"y={state.y:.4f} vy={state.vy:.4f} fx={fx:.4f} fy={fy:.4f}"
            )

    # Summary metrics in same style as previous analysis.
    i_peak = int(np.argmin(vy_hist))

    def _mean_window(lo: float, hi: float) -> float:
        m = (y_hist <= lo) & (y_hist >= hi)
        if not np.any(m):
            return float("nan")
        return float(np.mean(vy_hist[m]))

    summary = {
        "case_name": case_name,
        "dt": float(dt),
        "nt": int(nt),
        "heun": bool(heun),
        "mass": mass,
        "radius": radius,
        "inertia": inertia,
        "rho_fluid": rho_fluid,
        "rho_body": rho_body,
        "gravity": g,
        "water_surface": water_surface,
        "t_end": float(times[-1]),
        "x_end": float(x_hist[-1]),
        "y_end": float(y_hist[-1]),
        "vy_end": float(vy_hist[-1]),
        "peak_vy": float(vy_hist[i_peak]),
        "t_peak": float(times[i_peak]),
        "y_peak": float(y_hist[i_peak]),
        "vy_mean_y_1p5_1p0": _mean_window(1.5, 1.0),
        "vy_mean_y_1p2_0p8": _mean_window(1.2, 0.8),
        "vy_mean_y_1p0_0p6": _mean_window(1.0, 0.6),
        "vy_tail_mean": float(np.mean(vy_hist[int(0.9 * len(vy_hist)):])),
    }

    np.savez_compressed(
        case_dir / "adhoc_timeseries.npz",
        t=times,
        x=x_hist,
        y=y_hist,
        vx=vx_hist,
        vy=vy_hist,
        fx=fx_hist,
        fy=fy_hist,
        buoy=buoy_hist,
    )
    with (case_dir / "adhoc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {case_dir / 'adhoc_timeseries.npz'}")
    print(f"[saved] {case_dir / 'adhoc_summary.json'}")

    return summary


def run_default_study(t_end: float):
    rows = []
    for case in DEFAULT_CASES:
        rows.append(
            run_case(
                case_name=case["name"],
                dt=float(case["dt"]),
                t_end=t_end,
                heun=bool(case["heun"]),
            )
        )

    out_csv = OUT_ROOT / "adhoc_summary.csv"
    keys = [
        "case_name", "dt", "nt", "heun", "t_end", "y_end", "vy_end",
        "peak_vy", "t_peak", "y_peak",
        "vy_mean_y_1p5_1p0", "vy_mean_y_1p2_0p8", "vy_mean_y_1p0_0p6",
        "vy_tail_mean",
    ]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[saved] {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Standalone Coquerelle FSI (no MuJoCo)")
    parser.add_argument("--case-name", type=str, default=None,
                        help="Run a single case with this name")
    parser.add_argument("--dt", type=float, default=5.0e-5,
                        help="Time-step for single-case run")
    parser.add_argument("--t-end", type=float, default=0.35,
                        help="Physical end time [s]")
    parser.add_argument("--heun", action="store_true",
                        help="Use Heun for single-case run (default Euler)")
    parser.add_argument("--run-default-study", action="store_true",
                        help="Run default multi-case study")
    args = parser.parse_args()

    if args.run_default_study:
        run_default_study(t_end=args.t_end)
        return

    case_name = args.case_name
    if case_name is None:
        case_name = ("heun" if args.heun else "euler") + f"_dt_{args.dt:.8g}".replace(".", "p")

    run_case(
        case_name=case_name,
        dt=float(args.dt),
        t_end=float(args.t_end),
        heun=bool(args.heun),
    )


if __name__ == "__main__":
    main()
