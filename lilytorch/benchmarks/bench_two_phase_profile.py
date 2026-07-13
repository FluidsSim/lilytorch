#!/usr/bin/env python3
"""4.1 two-phase profiling: per-region timings for standard two-phase cases.

Measures the per-region wall-clock cost on the native (post-Phase-1) path
for two canonical cases:

* **surface-pool 2-D** — a body at/near the waterline in a 2-D domain
  (water below, air above), driven by gravity.
* **sphere water-entry 3-D** — a sphere falling through a water surface
  into a pool.

Each region is timed with CUDA events (GPU-synchronised) and reported as
ms/step averaged over a steady-state window.

Usage::

    python lilytorch/benchmarks/bench_two_phase_profile.py  [--dim 2|3]  [--dtype float32|float64]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import contextlib

import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from lilytorch.src.two_phase_solver import TwoPhaseSolver
from lilytorch.src.body import CompositeBodyAnalytical


# ---------------------------------------------------------------------------
# 2-D surface-pool config  (body straddling the waterline)
# ---------------------------------------------------------------------------
def _pars_2d_surface_pool(N=128, dtype=torch.float32, use_gpu=True):
    """2-D surface pool: water fills y < 0.5, air above, a cylinder at (0.5, 0.45)."""
    L = 1.0
    waterline = 0.5
    return {
        "solver": {
            "use_gpu": use_gpu,
            "nthreads": 1,
            "Nx": N, "Ny": N,
            "xmin": 0.0, "xmax": L,
            "ymin": 0.0, "ymax": L,
            "nt": 500,
            "nu": 1.0e-6,
            "rho": 1000.0,
            "dt": 0.002,
            "convection_method": "quick",
            "poisson_method": "mgcg",
            "poisson_tol": 1.0e-4,
            "jacobi_weight": 0.8,
            "poisson_max_cycles": 40,
            "poisson_max_mgcg_cycles": 10,
            "poisson_nsmoothing": 2,
            "poisson_verbose": False,
            "bdim_mu0_projection": True,
            "bdim_body_div_correction": True,
            "force_method": "eulerian",
            "gravity": [0.0, -9.81],
            "two_phase": {
                "alpha_init": f"lambda X, Y: (Y < {waterline}).double()",
                "rho_water": 1000.0,
                "rho_air": 1.0,
                "nu_water": 1.0e-6,
                "nu_air": 1.5e-5,
                "air_transparent_body": True,
            },
        },
        "boundary_conditions": {
            "BC_type_u": ["D", "D", "D", "D"],
            "BC_values_u": [0.0, 0.0, 0.0, 0.0],
            "BC_type_v": ["D", "D", "D", "D"],
            "BC_values_v": [0.0, 0.0, 0.0, 0.0],
        },
        "body": {
            "type": "composite_analytical",
            "plotting": False,
            "sdf": ["lambda x, y: (x - 0.5)**2 + (y - 0.45)**2 - 0.01"],
            "update_maps": [{
                "rotation": "lambda t: 0.0*t",
                "translation": ["lambda t: 0.0*t", "lambda t: 0.0*t"],
            }],
        },
        "output": {"save_frames": False, "save_every": 10**9},
    }


# ---------------------------------------------------------------------------
# 3-D sphere water-entry config
# ---------------------------------------------------------------------------
def _pars_3d_sphere_water_entry(N=48, dtype=torch.float32, use_gpu=True):
    """3-D sphere water entry: water fills z < 0.6, air above, sphere falls
    under gravity starting at z=0.75."""
    L = 1.0
    waterline = 0.6
    return {
        "solver": {
            "use_gpu": use_gpu,
            "nthreads": 1,
            "Nx": N, "Ny": N, "Nz": N,
            "xmin": 0.0, "xmax": L,
            "ymin": 0.0, "ymax": L,
            "zmin": 0.0, "zmax": L,
            "nt": 500,
            "nu": 1.0e-6,
            "rho": 1000.0,
            "dt": 0.002,
            "convection_method": "quick",
            "poisson_method": "mgcg",
            "poisson_tol": 1.0e-4,
            "jacobi_weight": 0.8,
            "poisson_max_cycles": 40,
            "poisson_max_mgcg_cycles": 10,
            "poisson_nsmoothing": 2,
            "poisson_verbose": False,
            "bdim_mu0_projection": True,
            "bdim_body_div_correction": True,
            "force_method": "eulerian",
            "gravity": [0.0, 0.0, -9.81],
            "two_phase": {
                "alpha_init": f"lambda X, Y, Z: (Z < {waterline}).double()",
                "rho_water": 1000.0,
                "rho_air": 1.0,
                "nu_water": 1.0e-6,
                "nu_air": 1.5e-5,
                "air_transparent_body": True,
            },
        },
        "boundary_conditions": {
            "BC_type_u": ["D"] * 6, "BC_values_u": [0.0] * 6,
            "BC_type_v": ["D"] * 6, "BC_values_v": [0.0] * 6,
            "BC_type_w": ["D"] * 6, "BC_values_w": [0.0] * 6,
        },
        "body": {
            "type": "composite_analytical",
            "plotting": False,
            "sdf": [
                "lambda x, y, z: (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.75)**2 - 0.01"
            ],
            "update_maps": [{
                "rotation": "lambda t: 0.0*t",
                "translation": [
                    "lambda t: 0.0*t",
                    "lambda t: 0.0*t",
                    "lambda t: -0.1*t",
                ],
            }],
        },
        "output": {"save_frames": False, "save_every": 10**9},
    }


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def cuda_timer(name, times_dict):
    """CUDA-event wall-clock timer for a named region."""
    if not torch.cuda.is_available():
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)  # ms
        times_dict.setdefault(name, []).append(elapsed)


# ---------------------------------------------------------------------------
# Profiled two-phase step (wraps solver to time each region)
# ---------------------------------------------------------------------------
def run_profiled_benchmark(pars, dtype, use_gpu, n_warmup=10, n_profile=100):
    """Run a two-phase simulation, profiling each region.

    Returns a dict mapping region name → list of ms timings.
    """
    solver = TwoPhaseSolver(pars, dtype=dtype, compute_forces=False)
    ndim = solver.ndim

    # Initialise velocities to zero + hydrostatic pressure
    if ndim == 2:
        u = torch.zeros_like(solver.u0)
        v = torch.zeros_like(solver.v0)
        p = torch.zeros_like(solver.p0)
    else:
        u = torch.zeros_like(solver.u0)
        v = torch.zeros_like(solver.v0)
        w = torch.zeros_like(solver.w0)
        p = torch.zeros_like(solver.p0)

    # --- Warm-up ---
    for it in range(n_warmup):
        t_step = it * float(solver.dt)
        if ndim == 2:
            solver.composite_body.update(t_step, it, dt=solver.dt)
            out = solver.fluid_step(u, v, p, solver.dt)
            u, v, p = out
            solver.finalize_step(u, v, p, it)
        else:
            solver.composite_body.update(t_step, it, dt=solver.dt)
            out = solver.fluid_step(u, v, w, p, solver.dt)
            u, v, w, p = out
            solver.finalize_step(u, v, p, it, w_vel=w)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- Profiled run: instrumented step ---
    times = {}
    for it in range(n_warmup, n_warmup + n_profile):
        t_step = it * float(solver.dt)

        # ── Body update ──
        with cuda_timer("body_update", times):
            solver.composite_body.update(t_step, it, dt=solver.dt)

        # ── Fluid step (pre-Poisson region is graphed, so we time the graph
        #    replay + projection + post-projection separately) ──
        if ndim == 2:
            # Pre-Poisson (graph replay or eager)
            with cuda_timer("pre_poisson", times):
                # The graph captures advection+diffusion+bdim_forcing+set_BCs
                # We can't easily split inside the graph, so time the whole
                # fluid_step minus projection
                pass

            # Full fluid_step (includes everything: pre-Poisson + projection)
            with cuda_timer("fluid_step", times):
                out = solver.fluid_step(u, v, p, solver.dt)
            u, v, p = out

            # cvof advection (VOF transport)
            with cuda_timer("cvof_advect", times):
                tp = solver.two_phase
                tp.advect(u, v, solver.dt)

        else:
            with cuda_timer("fluid_step", times):
                out = solver.fluid_step(u, v, w, p, solver.dt)
            u, v, w, p = out

            with cuda_timer("cvof_advect", times):
                tp = solver.two_phase
                tp.advect(u, v, w, solver.dt)

    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()

    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="4.1 two-phase profiling: per-region timings")
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3],
                    help="Spatial dimension (default: 2)")
    ap.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "float64"],
                    help="Floating-point precision (default: float32)")
    ap.add_argument("--N", type=int, default=None,
                    help="Grid size (default: 128 for 2D, 48 for 3D)")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Warm-up steps (default: 10)")
    ap.add_argument("--profile", type=int, default=100,
                    help="Profiled steps (default: 100)")
    args = ap.parse_args()

    ndim = args.dim
    N = args.N or (128 if ndim == 2 else 48)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    use_gpu = not args.no_gpu and torch.cuda.is_available()

    if ndim == 2:
        pars = _pars_2d_surface_pool(N=N, dtype=dtype, use_gpu=use_gpu)
        case_name = "2-D surface pool"
    else:
        pars = _pars_3d_sphere_water_entry(N=N, dtype=dtype, use_gpu=use_gpu)
        case_name = "3-D sphere water entry"

    print(f"=== 4.1 two-phase profile: {case_name} ===")
    print(f"  Grid: {N}^{ndim}, dtype: {args.dtype}, "
          f"device: {'gpu' if use_gpu else 'cpu'}")
    print(f"  Poisson: mgcg, tol=1e-4, mgcg_cycles=10, nsmoothing=2")
    print(f"  Warm-up: {args.warmup} steps, profile: {args.profile} steps")
    print()

    times = run_profiled_benchmark(
        pars, dtype, use_gpu,
        n_warmup=args.warmup, n_profile=args.profile)

    print(f"{'Region':<20s} {'mean (ms)':>10s} {'std (ms)':>10s} "
          f"{'min (ms)':>10s} {'max (ms)':>10s}")
    print("-" * 60)
    for name in sorted(times.keys()):
        arr = np.array(times[name])
        print(f"{name:<20s} {arr.mean():10.4f} {arr.std():10.4f} "
              f"{arr.min():10.4f} {arr.max():10.4f}")

    # Summary
    print()
    total_per_step = sum(np.array(v).mean() for v in times.values())
    print(f"  Total per step (sum of regions): {total_per_step:.4f} ms")
    for name in sorted(times.keys()):
        arr = np.array(times[name])
        pct = 100.0 * arr.mean() / total_per_step
        print(f"    {name:<18s} {arr.mean():8.4f} ms  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
