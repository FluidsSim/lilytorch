#!/usr/bin/env python3
"""Repeatable peak-GPU-memory benchmark for the standalone FluidSolver.

Measures persistent and peak CUDA allocation over a short run of a
representative 3-D case (``flow_past_sphere_3d``) on the *python* solver path
(the multigrid ``project()`` path it exercises is shared with kernel mode, so
this is the right baseline for the T1/T3 memory-reduction items).

Usage
-----
    python lilytorch/benchmarks/bench_memory.py
    python .../bench_memory.py --nx 192 --ny 96 --nz 96 --nt 20 --warmup 5
    python .../bench_memory.py --config path/to/other_3d.yaml --convection abdquickest

Reports (all in GiB):
  * post-construction allocated  — persistent solver state
  * peak during measured steps   — torch.cuda.max_memory_allocated
  * post-run allocated           — leak check (should ≈ construction)
The peak number is the figure to drive down with T3a/T3b/T2a; re-run after
each stage and compare.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lilytorch.src.solver import FluidSolver               # noqa: E402
from lilytorch.util.yaml_operations import yaml2pyobject   # noqa: E402

DEFAULT_CONFIG = os.path.join(
    REPO_ROOT, "lilytorch", "src", "configs", "flow_past_sphere_3d.yaml"
)


def _gib(nbytes: int) -> float:
    return nbytes / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--nx", type=int, default=None, help="override Nx")
    ap.add_argument("--ny", type=int, default=None, help="override Ny")
    ap.add_argument("--nz", type=int, default=None, help="override Nz")
    ap.add_argument("--nt", type=int, default=20, help="measured steps")
    ap.add_argument("--warmup", type=int, default=5,
                    help="steps before the peak counter is reset")
    ap.add_argument("--convection", default=None,
                    help="override convection_method (quick/abdquickest/...)")
    ap.add_argument("--poisson", default="multigrid",
                    help="poisson_method (multigrid/mgcg/fft)")
    ap.add_argument("--dtype", default=None, help="float32/float64")
    ap.add_argument("--compile-adv-diff", action="store_true",
                    help="enable torch.compile on the advection-diffusion solve")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark targets GPU peak memory.")
        sys.exit(1)

    pars = yaml2pyobject(args.config)
    s = pars["solver"]
    if args.nx is not None: s["Nx"] = args.nx
    if args.ny is not None: s["Ny"] = args.ny
    if args.nz is not None: s["Nz"] = args.nz
    if args.convection is not None: s["convection_method"] = args.convection
    if args.dtype is not None: s["dtype"] = args.dtype
    s["poisson_method"] = args.poisson
    s["compile_adv_diff"] = bool(args.compile_adv_diff)
    s["nt"] = args.warmup + args.nt
    # no I/O during the benchmark
    pars["output"]["save_frames"] = False
    pars["output"]["save"] = False
    pars["output"].pop("existing_folder", None)

    grid = (s["Nx"], s["Ny"], s.get("Nz"))
    print(f"config={os.path.basename(args.config)}  grid={grid}  "
          f"conv={s['convection_method']}  poisson={args.poisson}  "
          f"warmup={args.warmup}  measured={args.nt}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    solver = FluidSolver(pars, compute_forces=False)
    torch.cuda.synchronize()
    alloc_construct = torch.cuda.memory_allocated()
    print(f"  post-construction allocated : {_gib(alloc_construct):7.3f} GiB")

    # --- manual loop; reset the peak counter EACH measured step so the
    #     reported peak is a true steady-state per-step peak (not polluted by
    #     one-time warmup allocations: lazy MG pyramid / FFT plan / cuDNN ws) ---
    u, v, p = solver.u0, solver.v0, solver.p0
    w = solver.w0
    per_step_peak = 0
    for it in range(solver.nt):
        measured = it >= args.warmup
        if measured:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = it * solver.dt
        u, v, p, w = solver.advance_and_compute_loads(u, v, p, it, t, w_vel=w)
        solver.finalize_step(u, v, p, it, w_vel=w)
        if measured:
            torch.cuda.synchronize()
            per_step_peak = max(per_step_peak, torch.cuda.max_memory_allocated())

    alloc_end = torch.cuda.memory_allocated()
    print(f"  PEAK per measured step      : {_gib(per_step_peak):7.3f} GiB   <-- drive this down")
    print(f"  post-run allocated          : {_gib(alloc_end):7.3f} GiB")


if __name__ == "__main__":
    main()
