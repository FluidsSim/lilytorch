#!/usr/bin/env python3
"""GPU utilisation benchmark — compares baseline vs three optimisation modes.

Tests all three small-grid GPU-utilisation strategies:
  1. ``adv_diff_streams``  — parallel u/v/w advection on separate CUDA streams
  2. ``compile_project``   — torch.compile on the pressure-projection step
  3. ``use_cuda_graphs``   — CUDA graph capture of the adv-diff solve

Timing uses CUDA events (wall-clock accurate on GPU), memory uses
torch.cuda.max_memory_allocated().

Usage
-----
    # 2-D, small grid
    python bench_gpu_util.py --dim 2 --nx 64 --ny 64

    # 3-D, small grid
    python bench_gpu_util.py --dim 3 --nx 64 --ny 64 --nz 64

    # 3-D, medium grid (shows where streams help less)
    python bench_gpu_util.py --dim 3 --nx 128 --ny 128 --nz 128

    # Use abdquickest scheme (cuda_graphs will be skipped automatically)
    python bench_gpu_util.py --dim 2 --nx 128 --ny 128 --scheme abdquickest
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lilytorch.src.solver import FluidSolver            # noqa: E402
from lilytorch.util.yaml_operations import yaml2pyobject  # noqa: E402


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------
_DEFAULT_2D = os.path.join(REPO_ROOT, "lilytorch", "src", "configs",
                           "flow_past_circle_2d.yaml")
_DEFAULT_3D = os.path.join(REPO_ROOT, "lilytorch", "src", "configs",
                           "flow_past_sphere_3d.yaml")


def _gib(n): return n / 1024**3


def _make_config(dim, ncells, scheme, poisson):
    """Load the appropriate YAML and set grid cells to match domain aspect ratio.

    Uses *ncells* as Ny (the shorter dimension).  Nx (and Nz for 3-D) are
    scaled by the domain aspect ratio so grid spacing is uniform.
    """
    cfg_path = _DEFAULT_2D if dim == 2 else _DEFAULT_3D
    pars     = yaml2pyobject(cfg_path)
    s        = pars["solver"]

    # Compute aspect ratios from domain extents in the config
    lx = float(s.get("xmax", 3.0)) - float(s.get("xmin", -1.0))
    ly = float(s.get("ymax", 1.0)) - float(s.get("ymin", -1.0))
    ny = ncells
    nx = max(1, round(ncells * lx / ly))

    s["Ny"] = ny
    s["Nx"] = nx
    if dim == 3:
        lz = float(s.get("zmax", 1.0)) - float(s.get("zmin", -1.0))
        s["Nz"] = max(1, round(ncells * lz / ly))

    s["convection_method"] = scheme
    s["poisson_method"]    = poisson
    s["solver_method"]     = "python"
    s["save_frames"]       = False
    s["save"]              = False
    s["nt"]                = 1          # overridden in run_config
    pars.get("output", {}).update({"save_frames": False, "save": False})
    return pars, (nx, ny) if dim == 2 else (nx, ny, s["Nz"])


def run_config(base_pars, flags: dict, warmup: int, measured: int,
               label: str) -> dict:
    """Build a solver with *flags* overrides, time *measured* steps.

    Returns dict with keys: label, ms_mean, ms_std, ms_min, peak_mib,
    persistent_mib, speedup (filled in by caller).
    """
    pars = copy.deepcopy(base_pars)
    s    = pars["solver"]
    s.update(flags)
    s["nt"] = warmup + measured

    # disable I/O
    for key in ("save_frames", "save"):
        s[key] = False
    pars.get("output", {}).update({"save_frames": False, "save": False})

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t_build0 = time.perf_counter()
    solver   = FluidSolver(pars, compute_forces=False)
    torch.cuda.synchronize()
    t_build1 = time.perf_counter()

    persistent_mib = torch.cuda.memory_allocated() / 1024**2

    u, v, p = solver.u0.clone(), solver.v0.clone(), solver.p0.clone()
    w = solver.w0.clone() if solver.ndim == 3 else None

    # ---- warmup (un-timed; also triggers lazy CUDA graph capture) ----
    for it in range(warmup):
        t  = it * solver.dt
        u, v, p, w = solver.advance_and_compute_loads(u, v, p, it, t, w_vel=w)
        solver.finalize_step(u, v, p, it, w_vel=w)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ---- measured steps ----
    step_times_ms = []
    for it in range(measured):
        t  = (warmup + it) * solver.dt
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end   = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        u, v, p, w = solver.advance_and_compute_loads(u, v, p, warmup + it, t, w_vel=w)
        solver.finalize_step(u, v, p, warmup + it, w_vel=w)
        ev_end.record()
        torch.cuda.synchronize()
        step_times_ms.append(ev_start.elapsed_time(ev_end))

    peak_mib = torch.cuda.max_memory_allocated() / 1024**2

    import statistics
    ms_mean = statistics.mean(step_times_ms)
    ms_std  = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
    ms_min  = min(step_times_ms)

    del solver
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
        label=label,
        ms_mean=ms_mean,
        ms_std=ms_std,
        ms_min=ms_min,
        peak_mib=peak_mib,
        persistent_mib=persistent_mib,
        build_s=t_build1 - t_build0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim",     type=int,   default=2,  choices=[2, 3])
    ap.add_argument("--ncells",  type=int,   default=64,
                    help="Ny (short dimension); Nx/Nz scaled to match domain aspect ratio")
    ap.add_argument("--scheme",  type=str,   default="quick",
                    help="convection scheme (quick / abdquickest / van_leer / ...)")
    ap.add_argument("--poisson", type=str,   default="multigrid")
    ap.add_argument("--warmup",  type=int,   default=10,
                    help="un-timed warmup steps (triggers JIT/graph compilation)")
    ap.add_argument("--steps",   type=int,   default=20,
                    help="measured steps per configuration")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark targets GPU utilisation.")
        sys.exit(1)

    base, grid = _make_config(args.dim, args.ncells, args.scheme, args.poisson)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice  : {dev}")
    print(f"Grid    : {grid}  dim={args.dim}  scheme={args.scheme}  "
          f"poisson={args.poisson}")
    print(f"Warmup  : {args.warmup}  Measured: {args.steps}\n")

    configs = [
        ("baseline",       {}),
        ("streams",        {"adv_diff_streams": True}),
        ("compile_project",{"compile_project":  True}),
        ("cuda_graphs",    {"use_cuda_graphs":  True}),
        # combined: streams + compile (no graphs, they conflict)
        ("streams+compile",{"adv_diff_streams": True, "compile_project": True}),
    ]

    results = []
    for label, flags in configs:
        print(f"  Running [{label}] ...", end=" ", flush=True)
        try:
            r = run_config(base, flags, args.warmup, args.steps, label)
            results.append(r)
            print(f"{r['ms_mean']:.1f} ms/step  "
                  f"peak {r['peak_mib']:.0f} MiB")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(None)

    # ---- summary table ----
    baseline = next((r for r in results if r and r["label"] == "baseline"), None)
    print("\n" + "="*72)
    print(f"{'Config':<20} {'ms/step':>8} {'±':>6} {'min':>6} "
          f"{'speedup':>8} {'peak MiB':>10} {'Δmem':>8}")
    print("-"*72)
    for r in results:
        if r is None:
            continue
        speedup = (f"{baseline['ms_mean']/r['ms_mean']:.2f}×"
                   if baseline and r["label"] != "baseline" else "  1.00×")
        dmem    = (f"{r['peak_mib'] - baseline['peak_mib']:+.0f}"
                   if baseline and r["label"] != "baseline" else "  base")
        print(f"{r['label']:<20} {r['ms_mean']:>8.2f} "
              f"{r['ms_std']:>6.2f} {r['ms_min']:>6.2f} "
              f"{speedup:>8} {r['peak_mib']:>10.0f} {dmem:>8}")
    print("="*72)
    print("speedup >1 = faster than baseline;  Δmem = peak MiB change vs baseline")

    if baseline:
        print(f"\nBaseline build time: {baseline['build_s']:.2f}s  "
              f"persistent: {baseline['persistent_mib']:.0f} MiB")


if __name__ == "__main__":
    main()
