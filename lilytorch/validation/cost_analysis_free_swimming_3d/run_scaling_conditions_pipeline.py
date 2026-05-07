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
After substantial testing (see git history of this folder) the best
method is ``nbforces_opt``.  The pipeline therefore exposes only two
conditions: that production method, plus an unoptimised reference so
the speed-up versus the no-cropping / no-batching baseline remains
visible.

    nboff         : reference baseline, all narrow-band flags off
                    (no cropping, no batching).
    nbforces_opt  : production method — streaming SDF + fused forces +
                    union cropping + narrow-band batching, with the
                    rotation-CSE / uniform-grid trilinear optimisation
                    of ``streaming_sdf_min_rho_3d_multi``.

Output layout
-------------
    figures/scaling_conditions/
        nboff/
        nbforces_opt/
        cost_scaling_loglog_conditions.pdf
        cost_scaling_speedup_conditions.pdf

Usage
-----
    python run_scaling_conditions_pipeline.py

    python run_scaling_conditions_pipeline.py --conditions nboff,nbforces_opt

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
        "label": "baseline (all flags off — no cropping, no batching)",
        "short_label": "baseline",
        "flags": [],
        "linestyle": "--",
        "marker": "s",
        "color": "#37474f",
    },
    # Production method.  Kernel-level optimisation of
    # ``streaming_sdf_min_rho_3d_multi`` (rotation CSE + uniform-grid
    # trilinear, commit 722c4cf) baked into every ``--streaming_sdf_3d``
    # path.  After substantial testing (see git history) this is the
    # best of every method that was previously benchmarked, so it is
    # the only optimised condition the pipeline still exposes.
    "nbforces_opt": {
        "label": "streaming + fused forces + optimised SDF "
                 "(rotation CSE + uniform-grid trilinear)",
        "short_label": "production",
        "flags": [
            "--streaming_sdf_3d",
            "--streaming_forces_3d",
            "--force_shared_union",
            "--mu_normals_union",
            "--bdim_union",
            "--force_narrow_batch",
        ],
        "linestyle": "-",
        "marker": "*",
        "color": "#f9a825",
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
    """Load median TOTAL step (ms) per grid from a condition's CSV folder.

    For backward compatibility this returns ``{grid: total_ms}``.  Use
    :func:`_load_step_breakdown_ms` when residual / per-category sums
    are needed too.
    """
    return {grid: bd["total"]
            for grid, bd in _load_step_breakdown_ms(data_dir).items()}


# Prefix taxonomy mirrors plot_scaling.py: every prefix matches exactly
# one explicit leaf timer column, so summing them and subtracting from
# "TOTAL step" yields the residual (un-attributed per-step work).  The
# residual is what dominates the plateau on the per-condition log-log,
# so making it visible at the pipeline level is the whole point of
# item #4 of the plan.
_EXPLICIT_PREFIXES = (
    "1b",
    "2 ",
    "3a  ",
    "3b",
    "3c ",
    "3d",
    "3e",
    "3f",
    "4 ",
    "5 ",
    "6 ",
)


def _load_step_breakdown_ms(data_dir):
    """Per-grid step breakdown read from cost_perstep_*.csv (preferred)
    or cost_breakdown_*.csv (fallback).

    Returns ``{grid: {"total": ms, "explicit": ms, "residual": ms}}``.
    The explicit sum uses the same leaf-timer prefixes as plot_scaling.py
    so the residual definition is consistent across all four scripts.
    """
    records = {}
    for csv_path in glob.glob(os.path.join(data_dir, "cost_breakdown_*.csv")):
        base = os.path.basename(csv_path)
        match = re.search(r"cost_breakdown_(\d+)x(\d+)x(\d+)\.csv$", base)
        if not match:
            continue
        grid = tuple(int(match.group(i)) for i in range(1, 4))
        perstep_path = csv_path.replace("cost_breakdown_", "cost_perstep_")

        total_ms = None
        explicit_ms = None
        if os.path.exists(perstep_path):
            try:
                df_ps = pd.read_csv(perstep_path)
                if "used" in df_ps.columns:
                    df_ps = df_ps[df_ps["used"] != "discarded"]
                if len(df_ps) > 0 and "TOTAL step" in df_ps.columns:
                    total_ms = float(df_ps["TOTAL step"].median())
                    matching = [c for c in df_ps.columns
                                if c != "TOTAL step"
                                and any(c.startswith(p) for p in _EXPLICIT_PREFIXES)]
                    if matching:
                        explicit_ms = float(df_ps[matching].sum(axis=1).median())
                    else:
                        explicit_ms = 0.0
            except Exception:
                total_ms = None
                explicit_ms = None

        if total_ms is None:
            # Fallback: cost_breakdown_*.csv aggregates only.  Residual
            # cannot be reconstructed exactly from aggregates because
            # per-step medians don't sum like means; we therefore set
            # ``explicit`` to None and let the caller skip residual.
            with open(csv_path) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("component", "").strip() == "TOTAL step":
                        if row.get("median_ms"):
                            total_ms = float(row["median_ms"])
                        elif row.get("mean_ms"):
                            total_ms = float(row["mean_ms"])
                        break

        if total_ms is None:
            continue

        if explicit_ms is None:
            residual_ms = None
        else:
            residual_ms = max(total_ms - explicit_ms, 0.0)

        records[grid] = {
            "total":    total_ms,
            "explicit": explicit_ms,
            "residual": residual_ms,
        }
    return records


def _plot_combined_loglog(root_dir, condition_ids):
    condition_records = {}
    condition_breakdowns = {}
    for cond in condition_ids:
        cond_dir = os.path.join(root_dir, cond)
        breakdown = _load_step_breakdown_ms(cond_dir)
        if breakdown:
            condition_breakdowns[cond] = breakdown
            condition_records[cond] = {g: bd["total"] for g, bd in breakdown.items()}

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

    # Residual-vs-TOTAL plot is independent of "nboff" being present —
    # render it before the speed-up plot's early-return guard.
    _plot_residual_vs_total(root_dir, condition_ids, condition_breakdowns)

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


def _plot_residual_vs_total(root_dir, condition_ids, condition_breakdowns):
    """Combined residual-vs-total log-log overlay (item #4 of the plan).

    For every condition we draw two curves on the same axes:
      • TOTAL step (solid)         — what the pipeline already reports
      • Residual = TOTAL − Σ explicit categories (dashed, same colour) —
        the un-attributed per-step work that drives the low-N plateau.

    A flat residual that doesn't shrink with N is the visual signature
    of launch / Python / FARMS overhead, and is what items #1 and #2 of
    the plan are intended to fix.  Until apply_forces is timed under
    category ``6 `` (and 3d/3e/3f are surfaced), the residual will
    over-report; that is exactly why the plot-side attribution changes
    in item #3 land in the same PR as this combined plot.
    """
    have_residual = any(
        bd.get("residual") is not None
        for breakdown in condition_breakdowns.values()
        for bd in breakdown.values()
    )
    if not have_residual:
        print("  Skipping residual-vs-total plot: no per-step CSVs with "
              "category columns found.")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for cond in condition_ids:
        breakdown = condition_breakdowns.get(cond)
        if not breakdown:
            continue
        grids = sorted(breakdown, key=_grid_cells)
        cells = np.array([_grid_cells(g) for g in grids], dtype=float)
        totals = np.array([breakdown[g]["total"] for g in grids], dtype=float)
        style = CONDITION_SPECS[cond]
        ax.loglog(
            cells, totals,
            linestyle="-",
            marker=style["marker"], markersize=6,
            linewidth=1.8, color=style["color"], markeredgewidth=1.0,
            label=f"{style['short_label']} — TOTAL",
        )
        residuals = np.array(
            [breakdown[g]["residual"] if breakdown[g]["residual"] is not None
             else np.nan for g in grids],
            dtype=float,
        )
        # Drop non-positive residuals so loglog doesn't choke on them.
        mask = np.isfinite(residuals) & (residuals > 0)
        if mask.any():
            ax.loglog(
                cells[mask], residuals[mask],
                linestyle="--",
                marker=style["marker"], markersize=5,
                linewidth=1.2, color=style["color"], alpha=0.85,
                markerfacecolor="white", markeredgewidth=1.0,
                label=f"{style['short_label']} — residual",
            )

    # O(N) reference anchored at the smallest TOTAL value across conditions.
    all_totals = [(c, t)
                  for breakdown in condition_breakdowns.values()
                  for c, t in ((float(_grid_cells(g)), bd["total"])
                               for g, bd in breakdown.items())]
    if len(all_totals) >= 2:
        all_totals.sort()
        x_ref = np.array([all_totals[0][0], all_totals[-1][0]], dtype=float)
        y_ref = all_totals[0][1] * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e", linestyle=":",
                  linewidth=1.0, alpha=0.7, label="O(N) reference")

    ax.set_xlabel("Total cells  $N_x N_y N_z$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("TOTAL vs residual per condition  "
                 "(residual = un-attributed per-step work)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5, ncol=2)
    fig.tight_layout()
    out_path = os.path.join(root_dir, "cost_residual_vs_total_conditions.pdf")
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


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
    default="nboff,nbforces_opt",
    help="Comma-separated condition ids. Valid: nboff (baseline reference, no flags), "
         "nbforces_opt (production method).",
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