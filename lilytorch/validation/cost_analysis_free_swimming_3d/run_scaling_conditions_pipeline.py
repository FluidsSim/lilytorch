#!/usr/bin/env python3
"""
Run the full multi-grid scaling pipeline under multiple narrow-band modes.

This wrapper reuses ``run_multigrid_cost_analysis.py`` for the expensive
simulation + per-condition plotting step, then generates one combined
log-log figure across conditions so it is easy to see when narrow-band
cropping and batching pay off.

Default grid ladder
-------------------
The default ladder uses only power-of-two axes and brackets 1e8 cells:

    256x64x64      =   1,048,576
    256x128x64     =   2,097,152
    256x128x128    =   4,194,304
    512x128x128    =   8,388,608
    512x256x128    =  16,777,216
    1024x256x128   =  33,554,432
    1024x256x256   =  67,108,864
    1024x512x256   = 134,217,728

This is intended for large-memory GPUs (for example an RTX 5090 with
64 GB).  On smaller GPUs use ``--grids`` or ``--conditions`` to trim
the run.

Conditions
----------
    nboff   : baseline, all narrow-band flags off
    nbcrop  : cropping only (force_shared_union + mu_normals_union + bdim_union)
    nbbatch : batching only (force_narrow_batch + batched_sdf_3d)
    nbon    : all narrow-band flags on (cropping + batching)

Output layout
-------------
    figures/scaling_conditions/
        nboff/
        nbcrop/
        nbbatch/
        nbon/
        cost_scaling_loglog_conditions.pdf
        cost_scaling_speedup_conditions.pdf

Usage
-----
    python run_scaling_conditions_pipeline.py

    python run_scaling_conditions_pipeline.py --conditions nboff,nbcrop,nbon

    python run_scaling_conditions_pipeline.py --grids \
        256:64:64,256:128:64,256:128:128,512:128:128

    python run_scaling_conditions_pipeline.py --plot-only
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_SCRIPT = os.path.join(SCRIPT_DIR, "run_multigrid_cost_analysis.py")

DEFAULT_GRIDS = [
    (256,   64,  64),
    (256,  128,  64),
    (256,  128, 128),
    (512,  128, 128),
    (512,  256, 128),
    (1024, 256, 128),
    # (1024, 256, 256),
    # (1024, 512, 256),
]

CONDITION_SPECS = {
    "nboff": {
        "label": "baseline (all flags off)",
        "short_label": "baseline",
        "flags": [],
        "linestyle": "--",
        "marker": "s",
        "color": "#37474f",
    },
    "nbcrop": {
        "label": "cropping only",
        "short_label": "cropping",
        "flags": ["--force_shared_union", "--mu_normals_union", "--bdim_union"],
        "linestyle": "-.",
        "marker": "^",
        "color": "#1976d2",
    },
    "nbbatch": {
        "label": "batching only",
        "short_label": "batching",
        "flags": ["--force_narrow_batch", "--batched_sdf_3d"],
        "linestyle": ":",
        "marker": "D",
        "color": "#ef6c00",
    },
    "nbtri": {
        "label": "custom trilinear (no batch, no crop)",
        "short_label": "trilinear",
        "flags": ["--custom_trilinear_3d"],
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "v",
        "color": "#6a1b9a",
    },
    "nbtri_crop": {
        "label": "custom trilinear + cropping",
        "short_label": "tri + crop",
        "flags": [
            "--custom_trilinear_3d",
            "--force_shared_union",
            "--mu_normals_union",
            "--bdim_union",
            "--force_narrow_batch",
        ],
        "linestyle": (0, (5, 1)),
        "marker": "P",
        "color": "#ad1457",
    },
    "nbstream": {
        "label": "streaming fused-CUDA (Phase B)",
        "short_label": "stream",
        "flags": ["--streaming_sdf_3d"],
        "linestyle": (0, (1, 1)),
        "marker": "X",
        "color": "#00695c",
    },
    "nbstream_crop": {
        "label": "streaming fused-CUDA + cropping",
        "short_label": "stream + crop",
        "flags": [
            "--streaming_sdf_3d",
            "--force_shared_union",
            "--mu_normals_union",
            "--bdim_union",
            "--force_narrow_batch",
        ],
        "linestyle": (0, (3, 1, 1, 1, 1, 1)),
        "marker": "*",
        "color": "#004d40",
    },
    "nbforces": {
        "label": "streaming + fused forces (Phase D)",
        "short_label": "stream + forces",
        "flags": [
            "--streaming_sdf_3d",
            "--streaming_forces_3d",
            "--force_shared_union",
            "--mu_normals_union",
            "--bdim_union",
            "--force_narrow_batch",
        ],
        "linestyle": "-",
        "marker": "h",
        "color": "#b71c1c",
    },
    "nbon": {
        "label": "cropping + batching",
        "short_label": "crop + batch",
        "flags": ["--union_narrow_band"],
        "linestyle": "-",
        "marker": "o",
        "color": "#2e7d32",
    },
}


def _parse_grid_triplets(s):
    grids = []
    for triplet in s.split(","):
        parts = triplet.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid grid triplet '{triplet}'")
        grids.append(tuple(int(p) for p in parts))
    return grids


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _validate_grids(grid_list):
    if not grid_list:
        raise ValueError("No grids supplied")
    for nx, ny, nz in grid_list:
        if not (_is_power_of_two(nx) and _is_power_of_two(ny) and _is_power_of_two(nz)):
            raise ValueError(
                f"Grid {(nx, ny, nz)} is invalid: Nx, Ny, Nz must all be powers of 2"
            )


def _grid_tag(grid):
    return f"{grid[0]}x{grid[1]}x{grid[2]}"


def _grid_label(grid):
    return f"{grid[0]}×{grid[1]}×{grid[2]}"


def _grid_cells(grid):
    return grid[0] * grid[1] * grid[2]


def _grid_arg(grid_list):
    return ",".join(f"{nx}:{ny}:{nz}" for nx, ny, nz in grid_list)


def _load_total_step_ms(data_dir):
    records = {}
    for csv_path in glob.glob(os.path.join(data_dir, "cost_breakdown_*.csv")):
        base = os.path.basename(csv_path)
        match = re.search(r"cost_breakdown_(\d+)x(\d+)x(\d+)\.csv$", base)
        if not match:
            continue
        grid = tuple(int(match.group(i)) for i in range(1, 4))
        perstep_path = csv_path.replace("cost_breakdown_", "cost_perstep_")
        total_ms = None
        if os.path.exists(perstep_path):
            try:
                df_ps = pd.read_csv(perstep_path)
                if "used" in df_ps.columns:
                    df_ps = df_ps[df_ps["used"] != "discarded"]
                if len(df_ps) > 0 and "TOTAL step" in df_ps.columns:
                    total_ms = float(df_ps["TOTAL step"].median())
            except Exception:
                total_ms = None
        if total_ms is None:
            with open(csv_path) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("component", "").strip() == "TOTAL step":
                        if row.get("median_ms"):
                            total_ms = float(row["median_ms"])
                        else:
                            total_ms = float(row["mean_ms"])
                        break
        if total_ms is not None:
            records[grid] = total_ms
    return records


def _plot_combined_loglog(root_dir, condition_ids):
    condition_records = {}
    for cond in condition_ids:
        cond_dir = os.path.join(root_dir, cond)
        records = _load_total_step_ms(cond_dir)
        if records:
            condition_records[cond] = records

    if len(condition_records) < 1:
        print("\n  Skipping combined plots: no condition CSVs found.")
        return False

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    all_points = []
    for cond in condition_ids:
        records = condition_records.get(cond)
        if not records:
            continue
        grids = sorted(records, key=_grid_cells)
        cells = np.array([_grid_cells(grid) for grid in grids], dtype=float)
        total_ms = np.array([records[grid] for grid in grids], dtype=float)
        style = CONDITION_SPECS[cond]
        ax.loglog(
            cells,
            total_ms,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.8,
            color=style["color"],
            markeredgewidth=1.2,
            label=style["label"],
        )
        all_points.extend(zip(cells.tolist(), total_ms.tolist()))

    if len(all_points) >= 2:
        all_points.sort()
        x_ref = np.array([all_points[0][0], all_points[-1][0]], dtype=float)
        y_anchor = all_points[0][1]
        y_ref = y_anchor * x_ref / x_ref[0]
        ax.loglog(
            x_ref,
            y_ref,
            color="#9e9e9e",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label="O(N) reference",
        )

    ax.set_xlabel("Total cells  $N_x N_y N_z$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Cost scaling across narrow-band conditions")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    out_path = os.path.join(root_dir, "cost_scaling_loglog_conditions.pdf")
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Figure saved → {out_path}")

    if "nboff" not in condition_records:
        return True

    base = condition_records["nboff"]
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.4))
    for cond in condition_ids:
        if cond == "nboff":
            continue
        records = condition_records.get(cond)
        if not records:
            continue
        common = sorted(set(base) & set(records), key=_grid_cells)
        if not common:
            continue
        cells = np.array([_grid_cells(grid) for grid in common], dtype=float)
        speedup = np.array([base[grid] / records[grid] for grid in common], dtype=float)
        style = CONDITION_SPECS[cond]
        ax2.semilogx(
            cells,
            speedup,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.6,
            color=style["color"],
            label=style["label"],
        )
    ax2.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=1.0, label="break-even (1×)")
    ax2.set_xlabel("Total cells  $N_x N_y N_z$")
    ax2.set_ylabel(r"Speed-up  $T_\mathrm{off}\,/\,T_\mathrm{mode}$")
    ax2.set_title("Speed-up vs baseline across narrow-band conditions")
    ax2.legend(loc="best", framealpha=0.9, fontsize=8)
    fig2.tight_layout()
    out_path2 = os.path.join(root_dir, "cost_scaling_speedup_conditions.pdf")
    fig2.savefig(out_path2)
    fig2.savefig(out_path2.replace(".pdf", ".png"))
    plt.close(fig2)
    print(f"  Figure saved → {out_path2}")
    return True


parser = argparse.ArgumentParser(
    description="Run the full multi-grid scaling pipeline under multiple narrow-band conditions",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--grids",
    type=str,
    default=None,
    help="Comma-separated Nx:Ny:Nz triplets. Defaults to the built-in power-of-two ladder up to 1e8 scale.",
)
parser.add_argument(
    "--conditions",
    type=str,
    default="nboff,nbcrop,nbbatch,nbon,nbtri,nbtri_crop,nbstream,nbstream_crop,nbforces",
    help="Comma-separated condition ids. Valid: nboff, nbcrop, nbbatch, nbon, nbtri, nbtri_crop, nbstream, nbstream_crop, nbforces",
)
parser.add_argument("--n_steps", type=int, default=50)
parser.add_argument("--precompile", type=int, default=30)
parser.add_argument("--discard_first", type=int, default=3)
parser.add_argument("--save_every", type=int, default=9999)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument(
    "--out_dir",
    type=str,
    default=os.path.join(SCRIPT_DIR, "figures", "scaling_conditions"),
    help="Root output directory. Each condition gets its own subfolder.",
)
parser.add_argument("--continue-on-error", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--plot-only",
    action="store_true",
    help="Skip simulations and only regenerate the combined condition plots from existing CSVs.",
)
args = parser.parse_args()

if args.grids is None:
    grids = list(DEFAULT_GRIDS)
else:
    try:
        grids = _parse_grid_triplets(args.grids)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

try:
    _validate_grids(grids)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

grids.sort(key=_grid_cells)

conditions = [cond.strip() for cond in args.conditions.split(",") if cond.strip()]
for cond in conditions:
    if cond not in CONDITION_SPECS:
        print(f"ERROR: Unknown condition '{cond}'. Valid: {list(CONDITION_SPECS)}")
        sys.exit(1)

args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

print("\n" + "=" * 72)
print("  Multi-Condition Scaling Pipeline — Free-Swimming 1guilla (3-D)")
print("=" * 72)
print(f"  Conditions:  {', '.join(conditions)}")
print(f"  Grids:       {', '.join(_grid_label(grid) for grid in grids)}")
print(f"  Total cells: {', '.join(f'{_grid_cells(grid):,}' for grid in grids)}")
print(f"  Steps:       {args.n_steps} measured + {args.precompile} precompile per grid")
print(f"  Device:      {args.device.upper()}")
print(f"  Output:      {args.out_dir}")
print(f"  Timestamp:   {datetime.now().isoformat()}")
print("=" * 72)

if not os.path.isfile(RUNNER_SCRIPT):
    print(f"ERROR: runner script not found: {RUNNER_SCRIPT}")
    sys.exit(1)

python_exe = sys.executable
failed_conditions = []

if not args.plot_only:
    grid_arg = _grid_arg(grids)
    for index, cond in enumerate(conditions, start=1):
        spec = CONDITION_SPECS[cond]
        cond_out = os.path.join(args.out_dir, cond)
        os.makedirs(cond_out, exist_ok=True)
        cmd = [
            python_exe,
            RUNNER_SCRIPT,
            "--grids",
            grid_arg,
            "--n_steps",
            str(args.n_steps),
            "--precompile",
            str(args.precompile),
            "--discard_first",
            str(args.discard_first),
            "--save_every",
            str(args.save_every),
            "--device",
            args.device,
            "--out_dir",
            cond_out,
        ]
        cmd.extend(spec["flags"])
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"\n{'─' * 72}")
        print(f"  [{index}/{len(conditions)}]  Condition {cond}  —  {spec['label']}")
        print(f"{'─' * 72}")
        print(f"  CMD: {' '.join(cmd)}")

        if args.dry_run:
            continue

        proc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed_conditions.append(cond)
            print(f"\n  FAILED: condition {cond} exited with code {proc.returncode}")
            if not args.continue_on_error:
                break

if args.dry_run:
    print("\n  [DRY RUN] No simulations were executed.")
    sys.exit(0)

print(f"\n{'─' * 72}")
print("  Generating combined condition plots…")
print(f"{'─' * 72}")

combined_ok = _plot_combined_loglog(args.out_dir, conditions)

if failed_conditions:
    print(f"\n  WARNING: failed conditions: {', '.join(failed_conditions)}")

if not combined_ok and failed_conditions:
    sys.exit(1)
if failed_conditions:
    sys.exit(1)