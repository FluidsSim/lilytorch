#!/usr/bin/env python3
"""Run the shared multi-grid benchmark under multiple solver modes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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
    default_pipeline_dir,
    get_dimension_spec,
    grid_arg,
    grid_label,
    parse_grid_list,
    parse_modes,
    plot_mode_comparison,
)


parser = argparse.ArgumentParser(
    description="Shared solver-mode scaling pipeline for pinned 1guilla"
)
parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
parser.add_argument("--sim", type=str, default="pinned", choices=["pinned"])
parser.add_argument("--modes", type=str, default="python,kernel")
parser.add_argument("--grids", type=str, default=None)
parser.add_argument("--preset", type=str, default="medium")
parser.add_argument("--n_steps", type=int, default=20)
parser.add_argument("--precompile", type=int, default=30)
parser.add_argument("--settle_steps", type=int, default=5)
parser.add_argument("--discard_first", type=int, default=5)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--skip-condition-plots", action="store_true")
parser.add_argument("--continue-on-error", action="store_true")
parser.add_argument("--save_every", type=int, default=9999)
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
    modes = parse_modes(args.modes)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

if args.out_dir is None:
    args.out_dir = default_pipeline_dir(SCRIPT_DIR, spec)
args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

if args.grids is not None:
    grids = parse_grid_list(args.grids, spec.dim)
    grid_cli = ["--grids", grid_arg(grids)]
else:
    grids = list(spec.presets[args.preset])
    grid_cli = ["--preset", args.preset]

runner_script = os.path.join(SCRIPT_DIR, "run_multigrid_cost_analysis.py")
if not os.path.isfile(runner_script):
    print(f"ERROR: runner script not found: {runner_script}")
    sys.exit(1)

print("\n" + "=" * 72)
print(f"  Shared Scaling Pipeline - {spec.benchmark_label} ({spec.label})")
print("=" * 72)
print(f"  Modes:       {', '.join(modes)}")
if args.grids is not None:
    print(f"  Grids:       {', '.join(grid_label(grid) for grid in grids)}")
else:
    print(f"  Preset:      {args.preset}")
print(f"  Steps:       {args.n_steps} measured + {args.precompile} precompile + {args.settle_steps} settle")
print(f"  Solver:      {args.poisson_method}, dtype={args.dtype}")
print(f"  Device:      {args.device.upper()}")
print(f"  Output:      {args.out_dir}")
print(f"  Timestamp:   {datetime.now().isoformat()}")
print("=" * 72)

python_exe = sys.executable
failed_modes = []

if not args.plot_only:
    for mode in modes:
        condition_dir = os.path.join(args.out_dir, mode)
        os.makedirs(condition_dir, exist_ok=True)
        cmd = [
            python_exe,
            runner_script,
            "--dim", str(spec.dim),
            "--sim", args.sim,
            "--mode", mode,
            *grid_cli,
            "--n_steps", str(args.n_steps),
            "--precompile", str(args.precompile),
            "--settle_steps", str(args.settle_steps),
            "--discard_first", str(args.discard_first),
            "--device", args.device,
            "--out_dir", condition_dir,
            "--save_every", str(args.save_every),
            "--dtype", args.dtype,
            "--poisson_method", args.poisson_method,
            "--timestep", str(args.timestep),
            "--spawn_x", str(args.spawn_x),
            "--freq", str(args.freq),
            "--twl", str(args.twl),
            "--amp", str(args.amp),
        ]
        if args.Lx_fixed is not None:
            cmd.extend(["--Lx_fixed", str(args.Lx_fixed)])
        if args.skip_condition_plots:
            cmd.append("--skip-plots")
        if args.continue_on_error:
            cmd.append("--continue-on-error")

        print(f"\n  Running mode '{mode}'")
        print(f"  CMD: {' '.join(cmd)}")
        if args.dry_run:
            print("  [DRY RUN] Skipping execution.")
            continue

        proc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed_modes.append(mode)
            if not args.continue_on_error:
                print(f"\n  Mode '{mode}' failed with exit code {proc.returncode}")
                sys.exit(proc.returncode)

if args.dry_run:
    print("\n  [DRY RUN] No simulations were executed.")
    sys.exit(0)

plot_generated = plot_mode_comparison(args.out_dir, spec, modes)
if plot_generated:
    print(f"\n  Combined mode-comparison figures saved in {args.out_dir}")
else:
    print("\n  ERROR: No mode CSVs were found.")

if failed_modes:
    print(f"\n  Failed modes: {', '.join(failed_modes)}")
    sys.exit(1)

if not plot_generated:
    sys.exit(1)