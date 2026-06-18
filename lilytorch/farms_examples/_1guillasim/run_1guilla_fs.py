#!/usr/bin/env python3
"""Standalone caller for 1guilla experiments — select between two-phase and
one-fluid free-surface solvers without modifying BDIMhandler/FARMS.

Usage:
  python run_1guilla_fs.py two_phase    # Original TwoPhaseSolver
  python run_1guilla_fs.py free_surface # One-fluid FreeSurfaceSolver

The config is read from a YAML file (default: simulation_config.yaml in the
current directory) or can be built programmatically by editing this script.
"""

import os, sys, argparse, math, torch
torch.set_default_device("cuda")

from lilytorch.src.two_phase_solver import TwoPhaseSolver
from lilytorch.src.free_surface_solver import FreeSurfaceSolver


def build_config():
    """Build a minimal 1guilla-style config dict.
    Override this function or load from YAML for your specific experiment.
    """
    u_inlet = 0.215971
    return {
        "solver": {
            "use_gpu": True, "nthreads": 1,
            "Nx": 256, "Ny": 64, "Nz": 32,
            "xmin": -0.9, "xmax": 1.5,
            "ymin": -0.3, "ymax": 0.3,
            "zmin": -0.15, "zmax": 0.15,
            "nt": 5001, "nu": 1e-6, "rho": 1000.0, "dt": 0.0001,
            "convection_method": "abdquickest",
            "poisson_tol": 1.0e-5, "poisson_max_cycles": 30,
            "poisson_max_mgcg_cycles": 10, "poisson_nsmoothing": 5,
            "poisson_warm_start": True, "poisson_method": "multigrid",
            "poisson_smoother": "jacobi", "poisson_bc_type": "neumann",
            "poisson_folder": "lilytorch/data/",
            "dtype": "float32", "solver_method": "python",
            "rho_body": 800.0, "zero_pressure_inside": True,
            "gravity": [0.0, 0.0, -9.81],
            "compile_adv_diff": False,
            # ---- Two-phase VOF (required by both solvers) ----
            "two_phase": {
                "alpha_init": "lambda X, Y, Z: (Z < 0.0).double()",
                "rho_water": 1000.0, "rho_air": 1.0,
                "nu_water": 1e-6, "nu_air": 1.5e-5,
                "face_density": "harmonic",
            },
            # ---- Free-surface (only used by FreeSurfaceSolver) ----
            "free_surface": {
                "extend_iters": 10,
                "use_gfm_gradient": False,
            },
        },
        "boundary_conditions": {
            "BC_type_u": ["D", "D", "N", "N", "N", "N"],
            "BC_values_u": [u_inlet, u_inlet, 0.0, 0.0, 0.0, 0.0],
            "BC_type_v": ["D", "D", "N", "N", "N", "N"],
            "BC_values_v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "BC_type_w": ["D", "D", "N", "N", "N", "N"],
            "BC_values_w": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "body": {
            "type": "composite_mesh",
            "plotting": False,
            "sdf": [],
            "update_maps": [],
        },
        "output": {
            "save_path": "/tmp/1guilla_fs_test/",
            "save_frames": False, "save_every": 10**9,
            "save": False, "save_drags": False,
            "vmin": -1, "vmax": 1,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="1guilla free-surface caller")
    parser.add_argument("mode", nargs="?", default="free_surface",
                        choices=["two_phase", "free_surface"],
                        help="Solver mode (default: free_surface)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"])
    args = parser.parse_args()

    # Load config
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            pars = yaml.safe_load(f)
    else:
        pars = build_config()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Select solver class
    if args.mode == "free_surface":
        SolverCls = FreeSurfaceSolver
        # Ensure free_surface block exists
        pars["solver"].setdefault("free_surface", {"extend_iters": 10})
    else:
        SolverCls = TwoPhaseSolver

    print(f"[run_1guilla_fs] mode={args.mode}  solver={SolverCls.__name__}  "
          f"grid={pars['solver']['Nx']}x{pars['solver']['Ny']}x{pars['solver']['Nz']}")

    # Instantiate solver
    solver = SolverCls(pars, dtype=dtype, compute_forces=True)
    solver.inside = lambda *a, **k: True
    solver.set_initial_conditions()
    u, v, p = solver.u0, solver.v0, solver.p0
    w = solver.w0 if solver.ndim == 3 else None

    nt = pars["solver"]["nt"]
    dt = float(solver.dt)
    print(f"[run_1guilla_fs] Starting {nt} steps, dt={dt:.2e}")

    for it in range(nt):
        if solver.ndim == 2:
            u, v, p, stop = solver.step_(u, v, p, it, it * dt)
        else:
            u, v, w, p, stop = solver.step_(u, v, p, it, it * dt, w_vel=w)

        if stop:
            print(f"[run_1guilla_fs] STOP (explosion) at iteration {it}")
            break

        if it % 500 == 0:
            umax = float(torch.maximum(u.abs().max(), v.abs().max()))
            if w is not None:
                umax = max(umax, float(w.abs().max()))
            print(f"  it={it}/{nt}  |u|max={umax:.3e}", flush=True)

    print(f"[run_1guilla_fs] Done. {it+1} steps completed.")


if __name__ == "__main__":
    main()
