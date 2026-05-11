#!/usr/bin/env python3
"""Shared multi-grid cost analysis for pinned 1guilla in 2-D or 3-D."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lilytorch.validation.cost_analysis.common import (
    DEFAULT_DTYPE,
    DEFAULT_POISSON_METHOD,
    DEFAULT_SPAWN_X,
    DEFAULT_TIMESTEP,
    DX_REF,
    MIN_LX_FISH,
    default_results_dir,
    get_dimension_spec,
    grid_arg,
    grid_cells,
    grid_label,
    grid_tag,
    load_cost_record,
    parse_grid_list,
    plot_multigrid_summary,
    resolve_solver_mode,
)


parser = argparse.ArgumentParser(
    description="Shared multi-grid cost analysis for pinned 1guilla"
)
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--sim", type=str, default="pinned", choices=["pinned"])
parser.add_argument("--grids", type=str, default=None,
                    help="Comma-separated grids, e.g. 128:32 or 256:64:64")
parser.add_argument("--preset", type=str, default="medium")
parser.add_argument("--n_steps", type=int, default=20)
parser.add_argument("--precompile", type=int, default=30)
parser.add_argument("--settle_steps", type=int, default=5)
parser.add_argument("--discard_first", type=int, default=5)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--skip-plots", action="store_true")
parser.add_argument("--continue-on-error", action="store_true")
parser.add_argument("--save_every", type=int, default=9999)
parser.add_argument("--mode", type=str, default=None, choices=["python", "kernel"])
parser.add_argument("--use_kernels", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--no_kernels", action="store_true",
                    help="DEPRECATED: alias for --mode python.")
parser.add_argument("--streaming_sdf_2d", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--force_narrow_batch", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--force_shared_union", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--mu_normals_union", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--bdim_union", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--streaming_sdf_3d", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--streaming_forces_3d", action="store_true",
                    help="DEPRECATED: alias for --mode kernel.")
parser.add_argument("--Lx_fixed", type=float, default=None)
parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["float32", "float64"])
parser.add_argument("--poisson_method", type=str, default=DEFAULT_POISSON_METHOD,
                    choices=["multigrid", "mgcg", "fft"])
parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
parser.add_argument("--spawn_x", type=float, default=DEFAULT_SPAWN_X)
parser.add_argument("--freq", type=float, default=1.0)
parser.add_argument("--twl", type=float, default=0.571429 * 14)
parser.add_argument("--amp", type=float, default=15.0)
args = parser.parse_args()

spec = get_dimension_spec(args.dim)
if args.preset not in spec.presets:
    print(f"ERROR: Unknown preset '{args.preset}'. Available presets: {', '.join(sorted(spec.presets))}")
    sys.exit(1)

try:
    SOLVER_MODE = resolve_solver_mode(args)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

if args.out_dir is None:
    args.out_dir = default_results_dir(SCRIPT_DIR, spec)
args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

if args.grids is not None:
    grids = parse_grid_list(args.grids, spec.dim)
else:
    grids = list(spec.presets[args.preset])
grids.sort(key=grid_cells)

# When the user does not pin a physical domain explicitly, use a *per-grid*
# domain Lx = Nx * DX_REF so that dx stays fixed at DX_REF across all grids
# and the tank grows proportionally with Nx.  This keeps the BDIM smoothing
# band (eps ∝ dx) constant in grid cells across resolutions, which is correct
# for a performance scaling study.  Pass --Lx_fixed explicitly to override
# (e.g. for a grid-convergence study with a shared physical domain).
_lx_fixed_user = args.Lx_fixed is not None

single_run_script = os.path.join(SCRIPT_DIR, "run_cost_analysis.py")
if not os.path.isfile(single_run_script):
    print(f"ERROR: single-run script not found: {single_run_script}")
    sys.exit(1)

print("\n" + "=" * 72)
print(f"  Shared Multi-Grid Cost Analysis - {spec.benchmark_label} ({spec.label})")
print("=" * 72)
print(f"  Grids:       {', '.join(grid_label(grid) for grid in grids)}")
print(f"  Total cells: {', '.join(f'{grid_cells(grid):,}' for grid in grids)}")
print(
    f"  Steps:       {args.n_steps} measured + {args.precompile} precompile + {args.settle_steps} settle per grid"
)
print(f"  Solver:      {args.poisson_method}, dtype={args.dtype}, mode={SOLVER_MODE or 'default'}")
if _lx_fixed_user:
    print(f"  Domain:      Lx_fixed = {args.Lx_fixed:.3f} m  [user-specified, same domain for all grids]")
else:
    per_lx = [f"{grid_label(g)}→{max(g[0] * DX_REF, MIN_LX_FISH):.4f} m" for g in grids]
    print(f"  Domain:      per-grid Lx = max(Nx×DX_REF, {MIN_LX_FISH}) (dx≈{DX_REF:.6f} m)  [{', '.join(per_lx)}]")
print(f"  Device:      {args.device.upper()}")
print(f"  Output:      {args.out_dir}")
print(f"  Timestamp:   {datetime.now().isoformat()}")
print("=" * 72)

results = {}
failed_grids = []
python_exe = sys.executable

for index, grid in enumerate(grids):
    tag = grid_tag(grid)
    header = (
        f"\n{'-' * 72}\n"
        f"  [{index + 1}/{len(grids)}] Grid {grid_label(grid)} ({grid_cells(grid):,} cells)\n"
        f"{'-' * 72}"
    )
    print(header)

    cmd = [
        python_exe,
        single_run_script,
        "--dim", str(spec.dim),
        "--sim", args.sim,
        "--n_steps", str(args.n_steps),
        "--precompile", str(args.precompile),
        "--settle_steps", str(args.settle_steps),
        "--discard_first", str(args.discard_first),
        "--save_every", str(args.save_every),
        "--device", args.device,
        "--out_dir", args.out_dir,
        "--dtype", args.dtype,
        "--poisson_method", args.poisson_method,
        "--timestep", str(args.timestep),
        "--spawn_x", str(args.spawn_x),
        "--freq", str(args.freq),
        "--twl", str(args.twl),
        "--amp", str(args.amp),
    ]
    if _lx_fixed_user:
        cmd.extend(["--Lx_fixed", str(args.Lx_fixed)])
    else:
        # Fixed dx = DX_REF; domain scales with Nx, but at least MIN_LX_FISH
        # so the fish always fits inside the tank even for coarse grids.
        cmd.extend(["--Lx_fixed", str(max(grid[0] * DX_REF, MIN_LX_FISH))])
    if SOLVER_MODE is not None:
        cmd.extend(["--mode", SOLVER_MODE])
    for field, value in zip(spec.grid_fields, grid):
        cmd.extend([f"--{field}", str(value)])

    print(f"  CMD: {' '.join(cmd)}")

    if args.dry_run:
        print("  [DRY RUN] Skipping execution.")
        results[tag] = {"rc": 0, "elapsed": 0.0, "csv": "", "record": None}
        continue

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
            timeout=7200,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"\n  ERROR: Grid {tag} timed out after 2 hours")
        rc = -1
    except Exception as exc:
        print(f"\n  ERROR: Grid {tag} raised {type(exc).__name__}: {exc}")
        rc = -2

    elapsed = time.time() - t0
    csv_path = os.path.join(args.out_dir, f"cost_breakdown_{tag}.csv")
    perstep_path = os.path.join(args.out_dir, f"cost_perstep_{tag}.csv")
    record = load_cost_record(csv_path, perstep_path) if os.path.isfile(csv_path) else None
    if rc == 0 and record is None:
        print(f"\n  ERROR: Grid {tag} exited cleanly but produced no readable cost CSV")
        rc = -3
    results[tag] = {
        "rc": rc,
        "elapsed": elapsed,
        "csv": csv_path if os.path.isfile(csv_path) else "",
        "record": record,
    }

    if rc != 0:
        failed_grids.append(tag)
        print(f"\n  FAILED: Grid {tag} exited with code {rc} (elapsed {elapsed:.1f} s)")
        if not args.continue_on_error:
            print("  Aborting. Use --continue-on-error to skip failures.")
            break
    else:
        print(f"\n  OK: Grid {tag} completed in {elapsed:.1f} s")
        if record is not None:
            print(f"      Median TOTAL step -> {record['total']:.2f} ms")

if args.dry_run:
    print("\n  [DRY RUN] No simulations were executed.")
    sys.exit(0)

print("\n\n" + "=" * 72)
print("  MULTI-GRID SUMMARY")
print("=" * 72)
print(f"  {'Grid':<18s} {'Cells':>12s} {'Status':>12s} {'Wall-time':>12s} {'Step median':>14s}")
print("-" * 72)
for grid in grids:
    tag = grid_tag(grid)
    result = results.get(tag, {})
    status = "OK" if result.get("rc") == 0 else f"FAIL({result.get('rc', '?')})"
    wall_time = f"{result.get('elapsed', 0.0):.1f} s"
    record = result.get("record")
    step_text = f"{record['total']:.2f} ms" if record is not None else "-"
    print(
        f"  {grid_label(grid):<18s} {grid_cells(grid):12,d} {status:>12s} {wall_time:>12s} {step_text:>14s}"
    )
print("=" * 72)

if failed_grids:
    print(f"\n  Failed grids: {', '.join(failed_grids)}")

plot_generated = False
if not args.skip_plots:
    plot_records = {
        grid: results[grid_tag(grid)]["record"]
        for grid in grids
        if results.get(grid_tag(grid), {}).get("record") is not None
    }
    mode_tag = SOLVER_MODE or "default"
    plot_generated = plot_multigrid_summary(args.out_dir, spec, mode_tag, plot_records)
    if plot_generated:
        print(f"  Scaling figures saved in {args.out_dir}")
    else:
        print("  ERROR: No valid CSV records were produced.")

if failed_grids:
    sys.exit(1)

if not args.skip_plots and not plot_generated:
    sys.exit(1)