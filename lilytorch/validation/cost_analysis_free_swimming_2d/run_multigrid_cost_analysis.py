#!/usr/bin/env python3
"""
Multi-grid computational-cost analysis for the 2-D pinned 1guilla.

Runs ``run_cost_analysis.py`` at several grid resolutions in a single
invocation (each in an isolated subprocess), then generates combined
scaling figures.  Subprocess isolation is essential because:

  ▸ FARMS creates persistent MuJoCo contexts
  ▸ torch.compile CUDA-graph recordings cannot be cleanly reset
  ▸ monkey-patching global state must be fresh for each grid

Grid dimensions should be powers of 2 (multigrid solver requirement).
The 2-D domain uses a 4:1 x:y aspect ratio (Nx = 4·Ny) to match the
production simulation domain (Lx=2.4 m, Ly=0.3 m at dx=dy≈0.00234 m).

Usage
-----
    # Run all default grids (small → large)
    python run_multigrid_cost_analysis.py

    # Custom grids (comma-separated Nx:Ny pairs)
    python run_multigrid_cost_analysis.py --grids 128:32,256:64,512:128

    # Compare one solver mode across the grid ladder
    python run_multigrid_cost_analysis.py --mode kernel

    # Override step counts
    python run_multigrid_cost_analysis.py --n_steps 80 --precompile 40 --settle_steps 10

    # Dry-run to see commands without executing
    python run_multigrid_cost_analysis.py --dry-run
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_RUN_SCRIPT   = os.path.join(SCRIPT_DIR, "run_cost_analysis.py")
PLOT_SCALING_SCRIPT = os.path.join(SCRIPT_DIR, "plot_scaling.py")

# ── Pre-defined grid presets (Nx, Ny pairs) ──────────────────────────
# Aspect ratio ≈ 4:1 mirrors the production 2-D domain (Lx/Ly = 2.4/0.3=8,
# but Nx/Ny = 1024/128 = 8 at production).  We expose a range of
# Nx/Ny = 4 pairs for the default ladder since 8:1 forces very small Ny
# at low Nx, leading to under-resolved y-direction at coarse grids.
PRESETS = {
    "small": [
        (128,  32),   #     4,096
        (256,  64),   #    16,384
        (512, 128),   #    65,536
    ],
    "medium": [
        (256,   64),  #    16,384
        (512,  128),  #    65,536
        (1024, 256),  #   262,144
    ],
    "large": [
        (256,   64),  #    16,384
        (512,  128),  #    65,536
        (1024, 256),  #   262,144
        (2048, 512),  # 1,048,576
    ],
    "full": [
        (128,  32),   #     4,096
        (256,  64),   #    16,384
        (512, 128),   #    65,536
        (1024, 256),  #   262,144
        (2048, 512),  # 1,048,576
    ],
    "production": [
        (256,   64),  #    16,384
        (512,  128),  #    65,536
        (1024, 256),  #   262,144
    ],
}

parser = argparse.ArgumentParser(
    description="Multi-grid cost analysis for the 2-D pinned 1guilla",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"""\
Preset grids (--preset <name>):
  small       {PRESETS['small']}
  medium      {PRESETS['medium']}
  large       {PRESETS['large']}
  full        {PRESETS['full']}
  production  {PRESETS['production']}
""",
)
parser.add_argument(
    "--grids", type=str, default=None,
    help="Comma-separated Nx:Ny pairs, e.g. '128:32,256:64,512:128'")
parser.add_argument(
    "--preset", type=str, default="medium", choices=list(PRESETS.keys()),
    help="Use a predefined grid set (default: medium)")
parser.add_argument(
    "--n_steps", type=int, default=20,
    help="Measured steps per grid (default: 20)")
parser.add_argument(
    "--precompile", type=int, default=30,
    help="Pre-compilation steps per grid (default: 30)")
parser.add_argument(
    "--settle_steps", type=int, default=5,
    help="Untimed settle steps per grid after pre-compilation (default: 5)")
parser.add_argument(
    "--discard_first", type=int, default=3,
    help="Discard first N timed steps (default: 3)")
parser.add_argument(
    "--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument(
    "--out_dir", type=str, default=None,
    help="Output directory for CSVs and figures (default: figures/ here)")
parser.add_argument(
    "--dry-run", action="store_true",
    help="Print commands without executing")
parser.add_argument(
    "--skip-plots", action="store_true",
    help="Skip combined scaling plot generation")
parser.add_argument(
    "--continue-on-error", action="store_true",
    help="Continue to next grid if one fails")
parser.add_argument(
    "--save_every", type=int, default=9999)
parser.add_argument(
    "--mode", type=str, default=None, choices=["python", "kernel"],
    help="Solver mode to benchmark across the grid ladder.")
parser.add_argument(
    "--streaming_sdf_2d", action="store_true",
    help="DEPRECATED: alias for --mode kernel.")
args = parser.parse_args()


def _resolve_solver_mode(cli_args):
    if cli_args.mode == "python":
        if cli_args.streaming_sdf_2d:
            raise ValueError("--mode python conflicts with --streaming_sdf_2d")
        return "python"
    if cli_args.mode == "kernel":
        return "kernel"
    if cli_args.streaming_sdf_2d:
        return "kernel"
    return None


try:
    SOLVER_MODE = _resolve_solver_mode(args)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

if args.out_dir is None:
    args.out_dir = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(args.out_dir, exist_ok=True)

# ── Parse grid list ──────────────────────────────────────────────────
if args.grids is not None:
    grids = []
    for pair in args.grids.split(","):
        parts = pair.strip().split(":")
        if len(parts) != 2:
            print(f"ERROR: Invalid grid pair '{pair}'. Expected Nx:Ny (e.g. 512:128)")
            sys.exit(1)
        grids.append(tuple(int(p) for p in parts))
else:
    grids = PRESETS[args.preset]

# Sort by total cell count (ascending)
grids.sort(key=lambda g: g[0] * g[1])


# ═══════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════

total_cells = [nx * ny for nx, ny in grids]
grid_strs   = [f"{nx}×{ny}" for nx, ny in grids]

print("\n" + "=" * 72)
print("  Multi-Grid Cost Analysis — Pinned 1guilla (2-D)")
print("=" * 72)
print(f"  Grids:       {', '.join(grid_strs)}")
print(f"  Total cells: {', '.join(f'{c:,}' for c in total_cells)}")
print(f"  Steps:       {args.n_steps} measured + {args.precompile} precompile + "
    f"{args.settle_steps} settle per grid")
print(f"  Device:      {args.device.upper()}")
if SOLVER_MODE is not None:
    print(f"  Mode:        {SOLVER_MODE}")
print(f"  Output:      {args.out_dir}")
print(f"  Timestamp:   {datetime.now().isoformat()}")
print("=" * 72)

if not os.path.isfile(SINGLE_RUN_SCRIPT):
    print(f"\nERROR: Single-grid script not found: {SINGLE_RUN_SCRIPT}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Run each grid
# ═══════════════════════════════════════════════════════════════════════

results = {}
failed_grids = []
python_exe = sys.executable

for i, (nx, ny) in enumerate(grids):
    tag = f"{nx}x{ny}"
    n_cells = nx * ny
    header = (f"\n{'─' * 72}\n"
              f"  [{i + 1}/{len(grids)}]  Grid {nx}×{ny}  ({n_cells:,} cells)\n"
              f"{'─' * 72}")
    print(header)

    cmd = [
        python_exe, SINGLE_RUN_SCRIPT,
        "--Nx", str(nx),
        "--Ny", str(ny),
        "--n_steps",      str(args.n_steps),
        "--precompile",   str(args.precompile),
        "--settle_steps", str(args.settle_steps),
        "--discard_first", str(args.discard_first),
        "--save_every",   str(args.save_every),
        "--device",       args.device,
        "--out_dir",      args.out_dir,
    ]
    if SOLVER_MODE is not None:
        cmd.extend(["--mode", SOLVER_MODE])

    print(f"  CMD: {' '.join(cmd)}")

    if args.dry_run:
        print("  [DRY RUN] Skipping execution.")
        results[tag] = {"rc": 0, "elapsed": 0.0, "csv": ""}
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
        print(f"\n  ERROR: Grid {tag} timed out after 2 hours!")
        rc = -1
    except Exception as e:
        print(f"\n  ERROR: Grid {tag} raised {type(e).__name__}: {e}")
        rc = -2

    elapsed = time.time() - t0
    csv_path = os.path.join(args.out_dir, f"cost_breakdown_{tag}.csv")
    csv_ok = os.path.isfile(csv_path)

    results[tag] = {
        "rc": rc,
        "elapsed": elapsed,
        "csv": csv_path if csv_ok else "",
    }

    if rc != 0:
        failed_grids.append(tag)
        print(f"\n  FAILED: Grid {tag} exited with code {rc} "
              f"(elapsed {elapsed:.1f} s)")
        if not args.continue_on_error:
            print("  Aborting. Use --continue-on-error to skip failures.")
            break
    else:
        print(f"\n  OK: Grid {tag} completed in {elapsed:.1f} s")
        if csv_ok:
            print(f"      CSV → {csv_path}")

if args.dry_run:
    print("\n  [DRY RUN] No simulations were executed.")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 72)
print("  MULTI-GRID SUMMARY")
print("=" * 72)
print(f"  {'Grid':<20s} {'Cells':>12s} {'Status':>10s} "
      f"{'Wall-time':>12s} {'CSV':>6s}")
print("-" * 72)

import csv as csv_mod

for (nx, ny) in grids:
    tag = f"{nx}x{ny}"
    r = results.get(tag, {})
    cells = nx * ny
    status = "OK" if r.get("rc") == 0 else f"FAIL({r.get('rc', '?')})"
    wt = f"{r.get('elapsed', 0):.1f} s"
    has_csv = "yes" if r.get("csv") else "no"

    step_ms = ""
    csv_path = r.get("csv", "")
    if csv_path and os.path.isfile(csv_path):
        try:
            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    if row.get("component", "").strip() == "TOTAL step":
                        if "median_ms" in row:
                            step_ms = f"  step={float(row['median_ms']):.2f} ms"
                        else:
                            step_ms = f"  step={float(row['mean_ms']):.2f} ms"
                        break
        except Exception:
            pass

    grid_label = f"{nx}×{ny}"
    print(f"  {grid_label:<20s} {cells:>12,d} {status:>10s} "
          f"{wt:>12s} {has_csv:>6s}{step_ms}")

print("=" * 72)

total_wall = sum(r.get("elapsed", 0) for r in results.values())
print(f"\n  Total wall-time: {total_wall:.1f} s ({total_wall / 60:.1f} min)")

if failed_grids:
    print(f"\n  WARNING: {len(failed_grids)} grid(s) failed: "
          f"{', '.join(failed_grids)}")


# ═══════════════════════════════════════════════════════════════════════
# Generate combined scaling plots
# ═══════════════════════════════════════════════════════════════════════

if args.skip_plots:
    print("\n  Skipping combined plots (--skip-plots).")
    sys.exit(0 if not failed_grids else 1)

csv_files = glob.glob(os.path.join(args.out_dir, "cost_breakdown_*.csv"))
n_csvs = len(csv_files)

if n_csvs < 2:
    print(f"\n  Skipping scaling plots: need ≥ 2 CSVs, found {n_csvs}.")
    sys.exit(0 if not failed_grids else 1)

print(f"\n{'─' * 72}")
print(f"  Generating combined scaling plots from {n_csvs} CSVs…")
print(f"{'─' * 72}")


def _grid_cells_from_path(p):
    # Matches NxM (2-D) patterns; ignores NxMxK (3-D) files in the same dir.
    m = re.search(r"(\d+)x(\d+)(?!x\d)(?:\.csv|_)", os.path.basename(p))
    if m:
        return int(m.group(1)) * int(m.group(2))
    return 0


def _generate_inline_scaling_plots(out_dir, csv_files_list):
    """Fallback: generate scaling plots without plot_scaling.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Times New Roman", "DejaVu Serif"],
        "font.size":       10,
        "axes.labelsize":  11,
        "axes.titlesize":  12,
        "figure.dpi":      300,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid":       True,
        "grid.alpha":      0.3,
    })

    cats = {
        "Body update (SDF)":      ["1b"],
        "mu + normals":           ["2 "],
        "Convection & diffusion": ["3a  "],
        "BDIM meta-equation":     ["3b"],
        "Projection (pressure)":  ["3c "],
        "set_BCs (2d)":           ["3d"],
        "var-density coeffs":     ["3e"],
        "release BDIM":           ["3f"],
        "Forces":                 ["4 "],
        "FARMS (apply_forces)":   ["6 "],
    }
    _OTHER = "Other (residual)"
    cat_colours = {
        "Body update (SDF)":      "#26a69a",
        "mu + normals":           "#66bb6a",
        "Convection & diffusion": "#42a5f5",
        "BDIM meta-equation":     "#ab47bc",
        "Projection (pressure)":  "#ef5350",
        "set_BCs (2d)":           "#fbc02d",
        "var-density coeffs":     "#7cb342",
        "release BDIM":           "#bcaaa4",
        "Forces":                 "#ffa726",
        "FARMS (apply_forces)":   "#5c6bc0",
        _OTHER:                   "#90a4ae",
    }

    # Only load 2-D CSVs (NxM, not NxMxK)
    csv_files_list = [
        p for p in csv_files_list
        if re.search(r"cost_breakdown_\d+x\d+(?!x\d)", os.path.basename(p))
    ]
    csv_files_list = sorted(csv_files_list, key=_grid_cells_from_path)

    g_labels, g_cells = [], []
    cat_data = {c: [] for c in cats}
    cat_data[_OTHER] = []

    for csv_f in csv_files_list:
        m = re.search(r"cost_breakdown_(\d+)x(\d+)(?!x\d)", os.path.basename(csv_f))
        if not m:
            continue
        fnx, fny = int(m.group(1)), int(m.group(2))
        g_labels.append(f"{fnx}×{fny}")
        g_cells.append(fnx * fny)

        df = pd.read_csv(csv_f)
        total_row = df[df["component"] == "TOTAL step"]
        n_st = int(total_row["calls"].iloc[0]) if len(total_row) else 1
        total_step_s = float(total_row["total_s"].iloc[0]) if len(total_row) else 0.0

        perstep_f = csv_f.replace("cost_breakdown_", "cost_perstep_")
        df_ps = None
        if os.path.exists(perstep_f):
            try:
                df_ps_raw = pd.read_csv(perstep_f)
                if "used" in df_ps_raw.columns:
                    df_ps = df_ps_raw[df_ps_raw["used"] != "discarded"]
                else:
                    df_ps = df_ps_raw
                if len(df_ps) == 0:
                    df_ps = None
            except Exception:
                df_ps = None

        def _cat_ms_per_step(prefixes):
            if df_ps is not None:
                matching_cols = [c for c in df_ps.columns
                                 if any(c.startswith(p) for p in prefixes)
                                 and c != "TOTAL step"]
                if matching_cols:
                    return float(df_ps[matching_cols].sum(axis=1).median())
            mask = df["component"].apply(
                lambda comp, pfx=prefixes: any(comp.startswith(p) for p in pfx))
            mask &= df["component"] != "TOTAL step"
            return 1e3 * df.loc[mask, "total_s"].sum() / n_st

        def _total_ms_per_step():
            if df_ps is not None and "TOTAL step" in df_ps.columns:
                return float(df_ps["TOTAL step"].median())
            return 1e3 * total_step_s / n_st

        explicit_ms = 0.0
        for cat_name, prefixes in cats.items():
            cat_ms = _cat_ms_per_step(prefixes)
            explicit_ms += cat_ms
            cat_data[cat_name].append(cat_ms)

        total_ms = _total_ms_per_step()
        cat_data[_OTHER].append(max(total_ms - explicit_ms, 0.0))

    n = len(g_labels)
    if n < 2:
        return

    plot_order = list(cats.keys()) + [_OTHER]

    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n + 2), 4.0))
    x = np.arange(n)
    bottoms = np.zeros(n)
    for cat in plot_order:
        vals = np.array(cat_data[cat])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, 0.55, bottom=bottoms, label=cat,
               color=cat_colours.get(cat, "#bdbdbd"),
               edgecolor="white", linewidth=0.6)
        bottoms += vals
    for i in range(n):
        ax.text(x[i], bottoms[i] + 0.02 * bottoms.max(),
                f"{bottoms[i]:.1f} ms", ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels)
    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Cost breakdown – 2-D pinned 1guilla")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, bottoms.max() * 1.15)
    fig.tight_layout()
    fpath = os.path.join(out_dir, "cost_scaling_stacked.pdf")
    fig.savefig(fpath); fig.savefig(fpath.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Figure → {fpath}")


if os.path.isfile(PLOT_SCALING_SCRIPT):
    proc = subprocess.run(
        [python_exe, PLOT_SCALING_SCRIPT,
         "--data_dir", args.out_dir,
         "--out_dir",  args.out_dir],
        cwd=SCRIPT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if proc.returncode != 0:
        print(f"  WARNING: plot_scaling.py exited with code {proc.returncode}")
else:
    _generate_inline_scaling_plots(args.out_dir, csv_files)


# ═══════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Multi-grid cost analysis complete.                                ║
║                                                                    ║
║  CSVs + figures saved in: {args.out_dir:<42s} ║
║                                                                    ║
║  Key outputs:                                                      ║
║    cost_breakdown_<NxM>.csv      – per-grid timing breakdown       ║
║    cost_perstep_<NxM>.csv        – raw per-step timings            ║
║    cost_scaling_stacked.pdf      – stacked bar comparison          ║
║    cost_scaling_loglog.pdf       – log-log scaling trend           ║
║    cost_scaling_pct.pdf          – percentage distribution         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

sys.exit(0 if not failed_grids else 1)
