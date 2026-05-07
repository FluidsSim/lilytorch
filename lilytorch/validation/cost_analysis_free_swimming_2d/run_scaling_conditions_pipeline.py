#!/usr/bin/env python3
"""
Run the full multi-grid scaling pipeline under the two solver modes
for the 2-D pinned 1guilla.

Modes
-----
    python  : reference path with ``solver_method = "python"``.
    kernel  : optimised path with ``solver_method = "kernel"``.

Default grid ladder
-------------------
    128×32        =     4,096 cells
    256×64        =    16,384 cells
    512×128       =    65,536 cells
    1024×256      =   262,144 cells
    2048×512      = 1,048,576 cells

Output layout
-------------
    figures/scaling_conditions/
        python/
        kernel/
        cost_scaling_loglog_conditions.pdf
        cost_scaling_speedup_conditions.pdf
        cost_residual_vs_total_conditions.pdf

Usage
-----
    python run_scaling_conditions_pipeline.py

    python run_scaling_conditions_pipeline.py --modes python,kernel

    python run_scaling_conditions_pipeline.py --grids 256:64,512:128,1024:256

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

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RUNNER_SCRIPT = os.path.join(SCRIPT_DIR, "run_multigrid_cost_analysis.py")

DEFAULT_GRIDS = [
    (128,   32),
    (256,   64),
    (512,  128),
    (1024, 256),
    (2048, 512),
]

MODE_SPECS = {
    "python": {
        "label": "python mode (reference path)",
        "short_label": "python",
        "mode": "python",
        "linestyle": "--",
        "marker": "s",
        "color": "#37474f",
    },
    "kernel": {
        "label": "kernel mode (optimised path)",
        "short_label": "kernel",
        "mode": "kernel",
        "linestyle": "-",
        "marker": "*",
        "color": "#f9a825",
    },
}

MODE_ALIASES = {
    "nboff": "python",
    "nbforces_opt": "kernel",
}


def _parse_grid_pairs(s):
    grids = []
    for pair in s.split(","):
        parts = pair.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid grid pair '{pair}' (expected Nx:Ny)")
        grids.append(tuple(int(p) for p in parts))
    return grids


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _validate_grids(grid_list):
    if not grid_list:
        raise ValueError("No grids supplied")
    for nx, ny in grid_list:
        if not (_is_power_of_two(nx) and _is_power_of_two(ny)):
            raise ValueError(
                f"Grid ({nx}, {ny}) is invalid: Nx and Ny must be powers of 2"
            )


def _grid_tag(grid):
    return f"{grid[0]}x{grid[1]}"


def _grid_label(grid):
    return f"{grid[0]}×{grid[1]}"


def _grid_cells(grid):
    return grid[0] * grid[1]


def _grid_arg(grid_list):
    return ",".join(f"{nx}:{ny}" for nx, ny in grid_list)


# Prefix taxonomy — mirrors run_cost_analysis.py CATEGORIES.
# Each prefix matches exactly one explicit leaf timer column.
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
    "6 ",
)


def _load_step_breakdown_ms(data_dir):
    """Per-grid step breakdown from cost_perstep_*.csv (preferred) or
    cost_breakdown_*.csv (fallback).

    Only loads 2-D CSVs (NxM, not NxMxK).
    Returns ``{grid: {"total": ms, "explicit": ms, "residual": ms}}``.
    """
    records = {}
    # Match 2-D pattern NxM (no third dimension)
    for csv_path in glob.glob(os.path.join(data_dir, "cost_breakdown_*.csv")):
        base = os.path.basename(csv_path)
        match = re.search(r"cost_breakdown_(\d+)x(\d+)(?!x\d)", base)
        if not match:
            continue
        grid = (int(match.group(1)), int(match.group(2)))
        perstep_path = csv_path.replace("cost_breakdown_", "cost_perstep_")

        total_ms, explicit_ms = None, None
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
                    explicit_ms = float(df_ps[matching].sum(axis=1).median()) if matching else 0.0
            except Exception:
                total_ms = None

        if total_ms is None:
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

        residual_ms = max(total_ms - explicit_ms, 0.0) if explicit_ms is not None else None
        records[grid] = {
            "total":    total_ms,
            "explicit": explicit_ms,
            "residual": residual_ms,
        }
    return records


def _plot_combined_loglog(root_dir, condition_ids):
    condition_records    = {}
    condition_breakdowns = {}
    for cond in condition_ids:
        cond_dir = os.path.join(root_dir, cond)
        breakdown = _load_step_breakdown_ms(cond_dir)
        if breakdown:
            condition_breakdowns[cond] = breakdown
            condition_records[cond]    = {g: bd["total"] for g, bd in breakdown.items()}

    if not condition_records:
        print("\n  Skipping combined plots: no condition CSVs found.")
        return False

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    all_points = []
    for cond in condition_ids:
        records = condition_records.get(cond)
        if not records:
            continue
        grids    = sorted(records, key=_grid_cells)
        cells    = np.array([_grid_cells(g) for g in grids], dtype=float)
        total_ms = np.array([records[g]     for g in grids], dtype=float)
        style = MODE_SPECS[cond]
        ax.loglog(cells, total_ms,
                  linestyle=style["linestyle"], marker=style["marker"],
                  markersize=6, linewidth=1.8, color=style["color"],
                  markeredgewidth=1.2, label=style["label"])
        all_points.extend(zip(cells.tolist(), total_ms.tolist()))

    if len(all_points) >= 2:
        all_points.sort()
        x_ref   = np.array([all_points[0][0], all_points[-1][0]], dtype=float)
        y_ref   = all_points[0][1] * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e", linestyle=":",
                  linewidth=1.0, alpha=0.7, label="O(N) reference")

    ax.set_xlabel("Total cells  $N_x N_y$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Cost scaling across solver modes (2-D)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    out_path = os.path.join(root_dir, "cost_scaling_loglog_conditions.pdf")
    fig.savefig(out_path); fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Figure saved → {out_path}")

    _plot_residual_vs_total(root_dir, condition_ids, condition_breakdowns)

    if "python" not in condition_records:
        return True

    base = condition_records["python"]
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.4))
    for cond in condition_ids:
        if cond == "python":
            continue
        records = condition_records.get(cond)
        if not records:
            continue
        common  = sorted(set(base) & set(records), key=_grid_cells)
        if not common:
            continue
        cells   = np.array([_grid_cells(g) for g in common], dtype=float)
        speedup = np.array([base[g] / records[g] for g in common], dtype=float)
        style   = MODE_SPECS[cond]
        ax2.semilogx(cells, speedup,
                     linestyle=style["linestyle"], marker=style["marker"],
                     markersize=6, linewidth=1.6, color=style["color"],
                     label=style["label"])

    ax2.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=1.0,
                label="break-even (1×)")
    ax2.set_xlabel("Total cells  $N_x N_y$")
    ax2.set_ylabel(r"Speed-up  $T_\mathrm{python}\,/\,T_\mathrm{mode}$")
    ax2.set_title("Speed-up vs python baseline — 2-D solver modes")
    ax2.legend(loc="best", framealpha=0.9, fontsize=8)
    fig2.tight_layout()
    out_path2 = os.path.join(root_dir, "cost_scaling_speedup_conditions.pdf")
    fig2.savefig(out_path2); fig2.savefig(out_path2.replace(".pdf", ".png"))
    plt.close(fig2)
    print(f"  Figure saved → {out_path2}")
    return True


def _plot_residual_vs_total(root_dir, condition_ids, condition_breakdowns):
    """Combined residual-vs-TOTAL log-log overlay.

    Shows TOTAL step (solid) and un-attributed residual (dashed) per
    condition.  A flat residual reveals launch / Python / FARMS overhead
    driving the low-N plateau.
    """
    have_residual = any(
        bd.get("residual") is not None
        for breakdown in condition_breakdowns.values()
        for bd in breakdown.values()
    )
    if not have_residual:
        print("  Skipping residual-vs-total plot: no per-step CSVs found.")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for cond in condition_ids:
        breakdown = condition_breakdowns.get(cond)
        if not breakdown:
            continue
        grids   = sorted(breakdown, key=_grid_cells)
        cells   = np.array([_grid_cells(g) for g in grids], dtype=float)
        totals  = np.array([breakdown[g]["total"] for g in grids], dtype=float)
        style   = MODE_SPECS[cond]
        ax.loglog(cells, totals,
                  linestyle="-", marker=style["marker"], markersize=6,
                  linewidth=1.8, color=style["color"], markeredgewidth=1.0,
                  label=f"{style['short_label']} — TOTAL")

        residuals = np.array(
            [breakdown[g]["residual"] if breakdown[g]["residual"] is not None
             else np.nan for g in grids], dtype=float)
        mask = np.isfinite(residuals) & (residuals > 0)
        if mask.any():
            ax.loglog(cells[mask], residuals[mask],
                      linestyle="--", marker=style["marker"], markersize=5,
                      linewidth=1.2, color=style["color"], alpha=0.85,
                      markerfacecolor="white", markeredgewidth=1.0,
                      label=f"{style['short_label']} — residual")

    all_totals = [(float(_grid_cells(g)), bd["total"])
                  for breakdown in condition_breakdowns.values()
                  for g, bd in breakdown.items()]
    if len(all_totals) >= 2:
        all_totals.sort()
        x_ref = np.array([all_totals[0][0], all_totals[-1][0]], dtype=float)
        y_ref = all_totals[0][1] * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e", linestyle=":",
                  linewidth=1.0, alpha=0.7, label="O(N) reference")

    ax.set_xlabel("Total cells  $N_x N_y$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("TOTAL vs residual per condition  (2-D)\n"
                 "(residual = un-attributed per-step work)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.5, ncol=2)
    fig.tight_layout()
    out_path = os.path.join(root_dir, "cost_residual_vs_total_conditions.pdf")
    fig.savefig(out_path); fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run the full 2-D multi-grid scaling pipeline under both solver modes",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--grids", type=str, default=None,
    help="Comma-separated Nx:Ny pairs, e.g. '256:64,512:128,1024:256'")
parser.add_argument(
    "--modes", type=str, default="python,kernel",
    help="Comma-separated solver modes. Valid: python, kernel.")
parser.add_argument(
    "--conditions", type=str, default=None,
    help="DEPRECATED: alias for --modes. Legacy names nboff and nbforces_opt are mapped automatically.")
parser.add_argument("--n_steps",      type=int, default=20)
parser.add_argument("--precompile",   type=int, default=30)
parser.add_argument("--settle_steps", type=int, default=5)
parser.add_argument("--discard_first", type=int, default=3)
parser.add_argument("--save_every",   type=int, default=9999)
parser.add_argument("--device",       type=str, default="cuda",
                    choices=["cuda", "cpu"])
parser.add_argument(
    "--out_dir", type=str,
    default=os.path.join(SCRIPT_DIR, "figures", "scaling_conditions"),
    help="Root output directory. Each condition gets its own subfolder.")
parser.add_argument("--continue-on-error", action="store_true")
parser.add_argument("--dry-run",  action="store_true")
parser.add_argument(
    "--plot-only", action="store_true",
    help="Skip simulations and only regenerate combined condition plots from existing CSVs.")
args = parser.parse_args()

if args.grids is None:
    grids = list(DEFAULT_GRIDS)
else:
    try:
        grids = _parse_grid_pairs(args.grids)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

try:
    _validate_grids(grids)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

grids.sort(key=_grid_cells)

raw_modes = args.conditions if args.conditions is not None else args.modes
conditions = []
for raw in [c.strip() for c in raw_modes.split(",") if c.strip()]:
    cond = MODE_ALIASES.get(raw, raw)
    if cond not in MODE_SPECS:
        print(f"ERROR: Unknown mode '{raw}'. Valid: {list(MODE_SPECS)}")
        sys.exit(1)
    conditions.append(cond)

args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

print("\n" + "=" * 72)
print("  Multi-Mode Scaling Pipeline — Pinned 1guilla (2-D)")
print("=" * 72)
print(f"  Modes:       {', '.join(conditions)}")
print(f"  Grids:       {', '.join(_grid_label(g) for g in grids)}")
print(f"  Total cells: {', '.join(f'{_grid_cells(g):,}' for g in grids)}")
print(f"  Steps:       {args.n_steps} measured + {args.precompile} precompile + "
    f"{args.settle_steps} settle per grid")
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
        spec     = MODE_SPECS[cond]
        cond_out = os.path.join(args.out_dir, cond)
        os.makedirs(cond_out, exist_ok=True)
        cmd = [
            python_exe, RUNNER_SCRIPT,
            "--grids",         grid_arg,
            "--n_steps",       str(args.n_steps),
            "--precompile",    str(args.precompile),
            "--settle_steps",  str(args.settle_steps),
            "--discard_first", str(args.discard_first),
            "--save_every",    str(args.save_every),
            "--device",        args.device,
            "--out_dir",       cond_out,
            "--mode",          spec["mode"],
        ]
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"\n{'─' * 72}")
        print(f"  [{index}/{len(conditions)}]  Mode {cond}  —  {spec['label']}")
        print(f"{'─' * 72}")
        print(f"  CMD: {' '.join(cmd)}")

        if args.dry_run:
            continue

        proc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed_conditions.append(cond)
            print(f"\n  FAILED: mode {cond} exited with code {proc.returncode}")
            if not args.continue_on_error:
                break

if args.dry_run:
    print("\n  [DRY RUN] No simulations were executed.")
    sys.exit(0)

print(f"\n{'─' * 72}")
print("  Generating combined mode plots…")
print(f"{'─' * 72}")

combined_ok = _plot_combined_loglog(args.out_dir, conditions)

if failed_conditions:
    print(f"\n  WARNING: failed conditions: {', '.join(failed_conditions)}")

if not combined_ok and failed_conditions:
    sys.exit(1)
if failed_conditions:
    sys.exit(1)
