#!/usr/bin/env python3
"""Python-vs-GPU time split for the pressure-Poisson solve (native vs python).

The full coupled fluid step's native path needs the FARMS/BDIM streaming
bookkeeping to run, but the pressure Poisson solve is the dominant solver cost
and the subject of the "how much Python overhead?" question — so we profile it
directly on a representative variable-coefficient (BDIM immersed-body) operator.

For each backend (native CUDA kernel vs the Python reference loop) it reports,
per solve:
  * wall-clock  (perf_counter, synchronised)
  * GPU-busy    (sum of CUDA kernel durations from torch.profiler)
  * GPU-idle    (wall - busy = Python launch + CPU↔GPU sync stalls)
  * op launches (count of dispatched ops)

The gap between native and python wall-clock IS the Python overhead the kernels
were written to remove; the native GPU-idle fraction shows the residual
per-iteration .item() sync cost that even the native CG loop pays.

Usage:
    python bench_python_overhead.py --dim 3 --ncells 96 --poisson mgcg
    python bench_python_overhead.py --dim 2 --ncells 256
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lilytorch.src.poisson_mult import PoissonSolver       # noqa: E402


def make_problem(ndim, N, dtype, device, contrast=1000.0):
    L = 2 * math.pi
    h = L / N
    n = N + 2
    ax = torch.linspace(-h / 2, L + h / 2, n, dtype=dtype, device=device)
    grids = torch.meshgrid(*([ax] * ndim), indexing="ij")
    center = [0.5 * L] * ndim
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    mu0 = 0.5 * (1.0 + torch.tanh((torch.sqrt(r2) - 0.25 * L) / (2 * h)))
    c = (1.0 / contrast) + (1.0 - 1.0 / contrast) * mu0
    if ndim == 2:
        faces = {"ch": 0.5 * (c[1:, 1:-1] + c[:-1, 1:-1]),
                 "cv": 0.5 * (c[1:-1, 1:] + c[1:-1, :-1])}
    else:
        faces = {"ch": 0.5 * (c[1:, 1:-1, 1:-1] + c[:-1, 1:-1, 1:-1]),
                 "cv": 0.5 * (c[1:-1, 1:, 1:-1] + c[1:-1, :-1, 1:-1]),
                 "cw": 0.5 * (c[1:-1, 1:-1, 1:] + c[1:-1, 1:-1, :-1])}
    inner = tuple([slice(1, -1)] * ndim)
    f = torch.exp(-r2 / (2 * (0.15 * L) ** 2))[inner].clone()
    f -= f.mean()
    p0 = torch.zeros(*([n] * ndim), dtype=dtype, device=device)
    return h, faces, f, p0


def profile_backend(label, ndim, N, poisson, warmup, steps):
    dtype = torch.float32
    dev = "cuda"
    h, faces, f, p0 = make_problem(ndim, N, dtype, dev)
    ps = PoissonSolver(dtype, dev, h, tol=1e-5, max_cycles=50, max_vcycles=1,
                       nsmoothing=2, w=1.0, verbose=False, precond_vcycles=1,
                       smoother="rbgs")
    solve = ps.solve_mgcg if poisson == "mgcg" else ps.solve_multigrid

    for _ in range(warmup):
        solve(f, p0.clone(), **faces)
    torch.cuda.synchronize()

    # clean wall (headline speed, no profiler perturbation)
    t0 = time.perf_counter()
    for _ in range(steps):
        solve(f, p0.clone(), **faces)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3 / steps

    # profiler breakdown — measure wall and busy in the SAME (perturbed) run so
    # util = busy/wall is internally consistent (the profiler slows kernels, so
    # comparing profiler-busy to clean-wall would over-count and exceed 100%).
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    tp0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            solve(f, p0.clone(), **faces)
        torch.cuda.synchronize()
    prof_wall_ms = (time.perf_counter() - tp0) * 1e3 / steps
    ka = prof.key_averages()
    busy_ms = sum(k.self_device_time_total for k in ka) / 1e3 / steps
    launches = sum(k.count for k in ka) / steps
    util = 100.0 * busy_ms / prof_wall_ms
    return dict(label=label, wall=wall_ms, busy=busy_ms,
                idle=prof_wall_ms - busy_ms, util=util, launches=launches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3])
    ap.add_argument("--ncells", type=int, default=96)
    ap.add_argument("--poisson", type=str, default="mgcg",
                    choices=["mgcg", "multigrid"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)

    print(f"\nDevice  : {torch.cuda.get_device_name(0)}")
    print(f"Problem : {args.dim}-D N={args.ncells} variable-coeff (1000:1 jump), "
          f"{args.poisson}, fp32, rbgs")
    print(f"Warmup  : {args.warmup}   Measured: {args.steps}\n")

    rows = [
        profile_backend("native", args.dim, args.ncells,
                        args.poisson, args.warmup, args.steps),
    ]
    print("="*64)
    print(f"{'backend':<18}{'wall ms/solve':>14}{'GPU-util':>10}{'op launches':>14}")
    print("-"*64)
    for r in rows:
        print(f"{r['label']:<18}{r['wall']:>14.3f}{r['util']:>9.0f}%"
              f"{r['launches']:>14.0f}")
    print("="*64)
    print("  wall = clean perf_counter (no profiler);  GPU-util = kernel-busy /"
          " wall\n  measured under the profiler (same run);  launches incl."
          " C++-issued ATen ops")
    nat, py = rows
    print(f"\n  Python overhead removed by the kernel : "
          f"{py['wall'] - nat['wall']:.2f} ms/solve "
          f"({py['wall']/nat['wall']:.1f}× faster wall-clock)")
    print(f"  Native GPU-util {nat['util']:.0f}%  vs  Python GPU-util {py['util']:.0f}%"
          f"  →  the Python path leaves {100-py['util']:.0f}% of wall as launch"
          f" + sync idle")


if __name__ == "__main__":
    main()
