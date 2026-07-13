#!/usr/bin/env python3
"""0.4 gate: 600-step coupled parity + benchmark (cuda_native_port).

Runs a 2-D and 3-D coupled fluid simulation with an analytical body
for exactly 600 steps, records wall-clock ms/step (GPU-synchronised),
and saves the final state for parity comparison vs ``warp_port``.

Usage::

    python lilytorch/benchmarks/bench_04_gate.py  [--dim 2|3]  [--dtype float32|float64]

Report is printed to stdout; final state is saved as
``bench_04_gate_final_<dim>d.pt``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from lilytorch.src.solver import FluidSolver


def _pars(ndim: int, N: int, dt: float, nu: float, rho: float,
          body_sdf_lambda: str, use_gpu: bool = True) -> dict:
    L = 1.0
    solver = {
        "use_gpu": use_gpu,
        "nthreads": 1,
        "Nx": N, "Ny": N,
        "xmin": 0.0, "xmax": L,
        "ymin": 0.0, "ymax": L,
        "nt": 600,
        "nu": nu,
        "rho": rho,
        "dt": dt,
        "convection_method": "abdquickest",
        "poisson_method": "multigrid",
        "poisson_tol": 1.0e-12,
        "jacobi_weight": 0.8,
        "poisson_max_cycles": 40,
        "poisson_max_mgcg_cycles": 40,
        "poisson_nsmoothing": 6,
        "poisson_verbose": False,
        "bdim_mu0_projection": True,
        "bdim_body_div_correction": True,
        "force_method": "eulerian",
    }
    if ndim == 3:
        solver["Nz"] = N
        solver["zmin"] = 0.0
        solver["zmax"] = L
    n_faces = 2 * ndim
    bcs = {
        "BC_type_u": ["D"] * n_faces, "BC_values_u": [0.0] * n_faces,
        "BC_type_v": ["D"] * n_faces, "BC_values_v": [0.0] * n_faces,
    }
    if ndim == 3:
        bcs["BC_type_w"] = ["D"] * n_faces
        bcs["BC_values_w"] = [0.0] * n_faces
    body = {
        "type": "composite_analytical",
        "plotting": False,
        "sdf": [body_sdf_lambda],
        "update_maps": [{
            "rotation": "lambda t: 0.0*t",
            "translation": ["lambda t: 0.0*t"] * ndim,
        }],
    }
    output = {"save_frames": False, "save_every": 10**9, "vmin": -1.0, "vmax": 1.0}
    return {"solver": solver, "boundary_conditions": bcs,
            "body": body, "output": output}


def _taylor_green_ic(solver: FluidSolver):
    """Divergence-free Taylor–Green vortex (matches test_two_phase.py)."""
    two_pi = 2.0 * np.pi
    if solver.ndim == 2:
        X, Y = torch.meshgrid(solver.x, solver.y, indexing="ij")
        u =  torch.sin(two_pi * X) * torch.cos(two_pi * Y)
        v = -torch.cos(two_pi * X) * torch.sin(two_pi * Y)
        return u, v
    X, Y, Z = torch.meshgrid(solver.x, solver.y, solver.z, indexing="ij")
    u =  torch.sin(two_pi * X) * torch.cos(two_pi * Y)
    v = -torch.cos(two_pi * X) * torch.sin(two_pi * Y)
    w =  torch.zeros_like(u)
    return u, v, w


def _set_ic(solver: FluidSolver, fields):
    """Set initial conditions (matches test_two_phase.py)."""
    solver.set_initial_conditions()
    solver.u0 = fields[0].clone()
    solver.v0 = fields[1].clone()
    if solver.ndim == 3:
        solver.w0 = fields[2].clone()
    solver.p0 = torch.zeros_like(solver.u0)


def run_benchmark(ndim: int, N: int, dt: float, nu: float, rho: float,
                  dtype: torch.dtype, use_gpu: bool = True):
    """Run a 600-step coupled simulation and return timing + final state."""
    if ndim == 2:
        body_sdf = "lambda x, y: (x - 0.5)**2 + (y - 0.5)**2 - 0.04"
    else:
        body_sdf = "lambda x, y, z: (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2 - 0.04"

    pars = _pars(ndim, N, dt, nu, rho, body_sdf, use_gpu=use_gpu)
    solver = FluidSolver(pars, dtype=dtype, compute_forces=False)

    ic = _taylor_green_ic(solver)
    _set_ic(solver, ic)

    # Warm-up: 5 steps
    u, v, p = solver.u0, solver.v0, solver.p0
    w = solver.w0 if ndim == 3 else None
    for it in range(5):
        t_step = it * float(solver.dt)
        out = solver.advance_and_compute_loads(u, v, p, it, t_step, w_vel=w)
        u, v, p, w = out
        solver.finalize_step(u, v, p, it, w_vel=w)

    if use_gpu and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed run: 600 steps
    u, v, p = solver.u0, solver.v0, solver.p0
    w = solver.w0 if ndim == 3 else None

    times = []
    t_start = time.perf_counter()
    for it in range(5, 605):  # steps 5..604 = 600 steps
        t_step = it * float(solver.dt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = solver.advance_and_compute_loads(u, v, p, it, t_step, w_vel=w)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        u, v, p, w = out
        solver.finalize_step(u, v, p, it, w_vel=w)
        times.append((t1 - t0) * 1000.0)  # ms

    t_end = time.perf_counter()

    # Gather final state
    final_state = {"u": u.cpu().clone(), "v": v.cpu().clone(), "p": p.cpu().clone()}
    if ndim == 3 and w is not None:
        final_state["w"] = w.cpu().clone()

    ms_per_step = np.mean(times)
    ms_std = np.std(times)
    total_wall = t_end - t_start

    return {
        "ndim": ndim,
        "N": N,
        "dtype": str(dtype),
        "device": str(solver.device),
        "ms_per_step": ms_per_step,
        "ms_std": ms_std,
        "total_wall_s": total_wall,
        "n_steps": len(times),
        "final_state": final_state,
    }


def main():
    ap = argparse.ArgumentParser(description="0.4 gate: 600-step coupled parity + benchmark")
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3],
                    help="Spatial dimension (default: 2)")
    ap.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "float64"],
                    help="Floating-point precision (default: float32)")
    ap.add_argument("--N", type=int, default=None,
                    help="Grid size (default: 128 for 2D, 48 for 3D)")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    ndim = args.dim
    N = args.N or (128 if ndim == 2 else 48)
    dt = 0.004 if ndim == 2 else 0.002
    nu = 0.0001
    rho = 1000.0
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    use_gpu = not args.no_gpu and torch.cuda.is_available()

    print(f"=== 0.4 gate: {ndim}D coupled benchmark ===")
    print(f"  Grid: {N}^{ndim}, dtype: {args.dtype}, device: {'gpu' if use_gpu else 'cpu'}")
    print(f"  dt={dt}, nu={nu}, rho={rho}")
    print()

    result = run_benchmark(ndim, N, dt, nu, rho, dtype, use_gpu=use_gpu)

    print(f"  Steps:        {result['n_steps']}")
    print(f"  ms/step:      {result['ms_per_step']:.3f} ± {result['ms_std']:.3f}")
    print(f"  Total wall:   {result['total_wall_s']:.2f} s")
    print()

    # Save final state
    out_path = f"bench_04_gate_final_{ndim}d.pt"
    torch.save(result["final_state"], out_path)
    print(f"  Final state saved to: {out_path}")
    print(f"  u range: [{result['final_state']['u'].min():.6e}, {result['final_state']['u'].max():.6e}]")
    print(f"  v range: [{result['final_state']['v'].min():.6e}, {result['final_state']['v'].max():.6e}]")
    print(f"  p range: [{result['final_state']['p'].min():.6e}, {result['final_state']['p'].max():.6e}]")

    # Print checksums for parity comparison
    print()
    print("  --- Parity checksums (for warp_port comparison) ---")
    for key in result["final_state"]:
        t = result["final_state"][key]
        print(f"  {key}: sum={t.sum().item():.15e}  mean={t.mean().item():.15e}  "
              f"min={t.min().item():.15e}  max={t.max().item():.15e}")


if __name__ == "__main__":
    main()
