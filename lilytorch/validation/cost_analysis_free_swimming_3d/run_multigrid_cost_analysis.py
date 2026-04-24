#!/usr/bin/env python3
"""
Multi-grid computational-cost analysis for the 3-D free-swimming 1guilla.

Runs ``run_cost_analysis.py`` at several grid resolutions in a single
invocation (each in an isolated subprocess), then generates combined
scaling figures.  Subprocess isolation is essential because:

  ▸ FARMS creates persistent MuJoCo contexts
  ▸ torch.compile CUDA-graph recordings cannot be cleanly reset
  ▸ monkey-patching global state must be fresh for each grid

All grid dimensions must be powers of 2 (multigrid solver requirement).
The domain scales as N × dx_ref in each direction (dx_ref = 2.4/512),
so varying Nx:Ny:Nz ratios produces different domain shapes while
keeping h constant.  Minimum Nx = 128 (domain must contain the fish).

Usage
-----
    # Run all default grids (small → large)
    python run_multigrid_cost_analysis.py

    # Custom grids (comma-separated Nx:Ny:Nz triplets)
    python run_multigrid_cost_analysis.py --grids 128:32:32,256:64:64,512:128:128

    # Override step counts
    python run_multigrid_cost_analysis.py --n_steps 80 --precompile 40

    # Specific grids from presets
    python run_multigrid_cost_analysis.py --preset small   # 64→256
    python run_multigrid_cost_analysis.py --preset medium  # 128→512
    python run_multigrid_cost_analysis.py --preset large   # 256→1024

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
SINGLE_RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_cost_analysis.py")
PLOT_SCALING_SCRIPT = os.path.join(SCRIPT_DIR, "plot_scaling.py")

# ── Pre-defined grid presets ─────────────────────────────────────────
# All grid dimensions must be powers of 2 (multigrid requirement).
# The domain extents scale as N * dx_ref in each direction so that the
# grid spacing dx = 2.4/512 ≈ 0.0047 m is held CONSTANT across grids.
#
# Why the Nx = 128 grids are excluded from the default presets
# ------------------------------------------------------------
# For Nx < 256 the domain shrinks below the body length, so
# ``run_cost_analysis.py`` automatically falls back to a coarser dx
# (up to 1.8× production) so the fish still fits in Lx.  At that
# coarser resolution the BDIM-MuJoCo coupling becomes unstable over
# the ~ 120 step benchmark window (MuJoCo raises ``mjWARN_BADQACC``).
# Since the coarser grids would also measure a *different* physical
# problem, a fair scaling study keeps dx constant → only Nx ≥ 256.
PRESETS = {
    "small": [
        (256,  64,  64),       #  1,048,576
        (256, 128,  64),       #  2,097,152
        (256, 128, 128),       #  4,194,304
    ],
    "medium": [
        (256,  64,  64),       #  1,048,576
        (256, 128, 128),       #  4,194,304
        (512, 128, 128),       #  8,388,608
    ],
    "large": [
        (256, 128,  64),       #  2,097,152
        (256, 128, 128),       #  4,194,304
        (512, 128,  64),       #  4,194,304  (same N, diff shape)
        (512, 128, 128),       #  8,388,608
        (512, 256, 128),       # 16,777,216
    ],
    "full": [
        (256,  64,  64),       #  1,048,576
        (256, 128,  64),       #  2,097,152
        (256, 128, 128),       #  4,194,304
        (512, 128, 128),       #  8,388,608
        (512, 256, 128),       # 16,777,216
    ],
    "production": [
        (256,  64,  64),       #  1,048,576
        (256, 128,  64),       #  2,097,152
        (512, 128, 128),       #  8,388,608
        (512, 256, 128),       # 16,777,216
    ],
    # Legacy preset with coarse-dx 128^3 grids.  Kept for backwards
    # compatibility but expected to FAIL on the 128^x entries due to
    # the MuJoCo physics instability described above.
    "coarse_unstable": [
        (128,  32,  32),       #    131,072
        (128,  64,  32),       #    262,144
        (128,  64,  64),       #    524,288
        (256,  64,  64),       #  1,048,576
        (256, 128,  64),       #  2,097,152
        (256, 128, 128),       #  4,194,304
        (512, 128, 128),       #  8,388,608
        (512, 256, 128),       # 16,777,216
    ],
}

parser = argparse.ArgumentParser(
    description="Multi-grid cost analysis for the free-swimming 1guilla",
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
    help="Comma-separated Nx:Ny:Nz triplets, e.g. '64:16:16,128:32:32,256:64:64'")
parser.add_argument(
    "--preset", type=str, default="medium", choices=list(PRESETS.keys()),
    help="Use a predefined grid set (default: medium)")
parser.add_argument(
    "--n_steps", type=int, default=50,
    help="Measured steps per grid (default: 50)")
parser.add_argument(
    "--precompile", type=int, default=30,
    help="Pre-compilation steps per grid (default: 30)")
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
    "--save_every", type=int, default=9999,
    help="Save interval (default: 9999 = effectively never)")
parser.add_argument(
    "--force_shared_union", action="store_true",
    help="Crop shared-stress kernel to union of body AABBs (big win at ≥512³)")
parser.add_argument(
    "--force_narrow_batch", action="store_true",
    help="Use padded-batched narrow-band forces kernel")
parser.add_argument(
    "--mu_normals_union", action="store_true",
    help="Crop mu/normals kernel to union of body AABBs")
parser.add_argument(
    "--bdim_union", action="store_true",
    help="Crop BDIM2 meta-equation kernel to union of body AABBs")
parser.add_argument(
    "--batched_sdf_3d", action="store_true",
    help="Batched grid_sample SDF update (replaces per-body loop)")
parser.add_argument(
    "--union_narrow_band", action="store_true",
    help="Meta-flag: enables ALL narrow-band optimisations simultaneously "
         "(force_shared_union + mu_normals_union + bdim_union + "
         "batched_sdf_3d + force_narrow_batch).")
args = parser.parse_args()

if args.union_narrow_band:
    args.force_shared_union = True
    args.mu_normals_union = True
    args.bdim_union = True
    args.batched_sdf_3d = True
    args.force_narrow_batch = True

if args.out_dir is None:
    args.out_dir = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(args.out_dir, exist_ok=True)

# ── Parse grid list ──────────────────────────────────────────────────
if args.grids is not None:
    grids = []
    for triplet in args.grids.split(","):
        parts = triplet.strip().split(":")
        if len(parts) != 3:
            print(f"ERROR: Invalid grid triplet '{triplet}'. "
                  f"Expected Nx:Ny:Nz (e.g. 128:32:32)")
            sys.exit(1)
        grids.append(tuple(int(p) for p in parts))
else:
    grids = PRESETS[args.preset]

# Sort by total cell count (ascending)
grids.sort(key=lambda g: g[0] * g[1] * g[2])


# ═══════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════

total_cells = [nx * ny * nz for nx, ny, nz in grids]
grid_strs = [f"{nx}×{ny}×{nz}" for nx, ny, nz in grids]

print("\n" + "=" * 72)
print("  Multi-Grid Cost Analysis — Free-Swimming 1guilla (3-D)")
print("=" * 72)
print(f"  Grids:       {', '.join(grid_strs)}")
print(f"  Total cells: {', '.join(f'{c:,}' for c in total_cells)}")
print(f"  Steps:       {args.n_steps} measured + {args.precompile} precompile per grid")
print(f"  Device:      {args.device.upper()}")
print(f"  Output:      {args.out_dir}")
print(f"  Timestamp:   {datetime.now().isoformat()}")
print("=" * 72)

if not os.path.isfile(SINGLE_RUN_SCRIPT):
    print(f"\nERROR: Single-grid script not found: {SINGLE_RUN_SCRIPT}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Run each grid
# ═══════════════════════════════════════════════════════════════════════

results = {}             # grid_tag → {"rc": int, "elapsed": float, "csv": str}
failed_grids = []
python_exe = sys.executable

for i, (nx, ny, nz) in enumerate(grids):
    tag = f"{nx}x{ny}x{nz}"
    n_cells = nx * ny * nz
    header = (f"\n{'─' * 72}\n"
              f"  [{i + 1}/{len(grids)}]  Grid {nx}×{ny}×{nz}  "
              f"({n_cells:,} cells)\n"
              f"{'─' * 72}")
    print(header)

    cmd = [
        python_exe, SINGLE_RUN_SCRIPT,
        "--Nx", str(nx),
        "--Ny", str(ny),
        "--Nz", str(nz),
        "--n_steps", str(args.n_steps),
        "--precompile", str(args.precompile),
        "--discard_first", str(args.discard_first),
        "--save_every", str(args.save_every),
        "--device", args.device,
        "--out_dir", args.out_dir,
    ]
    if args.force_shared_union:
        cmd.append("--force_shared_union")
    if args.force_narrow_batch:
        cmd.append("--force_narrow_batch")
    if args.mu_normals_union:
        cmd.append("--mu_normals_union")
    if args.bdim_union:
        cmd.append("--bdim_union")
    if args.batched_sdf_3d:
        cmd.append("--batched_sdf_3d")

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
            timeout=7200,   # 2-hour timeout per grid
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

for (nx, ny, nz) in grids:
    tag = f"{nx}x{ny}x{nz}"
    r = results.get(tag, {})
    cells = nx * ny * nz
    status = "OK" if r.get("rc") == 0 else f"FAIL({r.get('rc', '?')})"
    wt = f"{r.get('elapsed', 0):.1f} s"
    has_csv = "yes" if r.get("csv") else "no"

    # Try to read the mean step time from the CSV
    step_ms = ""
    csv_path = r.get("csv", "")
    if csv_path and os.path.isfile(csv_path):
        try:
            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    if row.get("component", "").strip() == "TOTAL step":
                        # Prefer median (robust to CUDA tail outliers);
                        # fall back to mean for older CSVs that pre-date
                        # the median column.
                        if "median_ms" in row:
                            step_ms = f"  step={float(row['median_ms']):.2f} ms"
                        else:
                            step_ms = f"  step={float(row['mean_ms']):.2f} ms"
                        break
        except Exception:
            pass

    grid_label = f"{nx}×{ny}×{nz}"
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

# Check how many CSVs we have
csv_files = glob.glob(os.path.join(args.out_dir, "cost_breakdown_*.csv"))
n_csvs = len(csv_files)

if n_csvs < 2:
    print(f"\n  Skipping scaling plots: need ≥ 2 CSVs, found {n_csvs}.")
    sys.exit(0 if not failed_grids else 1)

print(f"\n{'─' * 72}")
print(f"  Generating combined scaling plots from {n_csvs} CSVs…")
print(f"{'─' * 72}")

def _grid_cells_from_path(p):
    m = re.search(r"(\d+)x(\d+)x(\d+)", os.path.basename(p))
    return int(m.group(1)) * int(m.group(2)) * int(m.group(3)) if m else 0


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
        # "3c " (trailing space) matches only the outer projection leaf
        # and NOT the nested "3c.i Jacobi" / "3c.ii V-cycle" sub-timers,
        # which would otherwise double-count projection on large grids.
        "Projection (pressure)":  ["3c "],
        "Forces":                 ["4 "],
        "Plotting & saving":      ["5 "],
        "FARMS (apply_forces)":   ["6 "],
    }
    _OTHER = "Other (residual)"
    cat_colours = {
        "Body update (SDF)":      "#26a69a",
        "mu + normals":           "#66bb6a",
        "Convection & diffusion": "#42a5f5",
        "BDIM meta-equation":     "#ab47bc",
        "Projection (pressure)":  "#ef5350",
        "Forces":                 "#ffa726",
        "Plotting & saving":      "#8d6e63",
        "FARMS (apply_forces)":   "#5c6bc0",
        _OTHER:                   "#90a4ae",
    }

    csv_files_list = sorted(csv_files_list, key=_grid_cells_from_path)
    g_labels, g_cells = [], []
    cat_data = {c: [] for c in cats}
    cat_data[_OTHER] = []

    for csv_f in csv_files_list:
        m = re.search(r"(\d+)x(\d+)x(\d+)", os.path.basename(csv_f))
        if not m:
            continue
        fnx, fny, fnz = int(m.group(1)), int(m.group(2)), int(m.group(3))
        g_labels.append(f"{fnx}×{fny}×{fnz}")
        g_cells.append(fnx * fny * fnz)

        df = pd.read_csv(csv_f)
        total_row = df[df["component"] == "TOTAL step"]
        n_st = int(total_row["calls"].iloc[0]) if len(total_row) else 1
        total_step_s = (
            float(total_row["total_s"].iloc[0]) if len(total_row) else 0.0
        )

        # Prefer per-step medians over mean-from-total_s when the
        # matching cost_perstep_*.csv is available.  Per-step medians
        # are robust to the single-recompile outliers that can inflate
        # dynamic-shape compiled kernel means by orders of magnitude.
        perstep_f = csv_f.replace("cost_breakdown_", "cost_perstep_")
        df_ps = None
        if os.path.exists(perstep_f):
            try:
                df_ps_raw = pd.read_csv(perstep_f)
                # Keep only measured (non-discarded) rows if the "used"
                # column is present.
                if "used" in df_ps_raw.columns:
                    df_ps = df_ps_raw[df_ps_raw["used"] != "discarded"]
                else:
                    df_ps = df_ps_raw
                if len(df_ps) == 0:
                    df_ps = None
            except Exception:
                df_ps = None

        def _cat_ms_per_step(prefixes):
            """Return category cost in ms/step, using per-step medians
            when available (robust to compile-outlier spikes), else
            falling back to mean from total_s / n_calls."""
            if df_ps is not None:
                # Sum all matching columns per step, then take median.
                matching_cols = [c for c in df_ps.columns
                                 if any(c.startswith(p) for p in prefixes)
                                 and c != "TOTAL step"]
                if matching_cols:
                    per_step_sum = df_ps[matching_cols].sum(axis=1)
                    return float(per_step_sum.median())
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
        residual_ms = max(total_ms - explicit_ms, 0.0)
        cat_data[_OTHER].append(residual_ms)

    n = len(g_labels)
    if n < 2:
        return

    plot_order = list(cats.keys()) + [_OTHER]

    # Stacked bar
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
    ax.set_title("Cost breakdown – free-swimming 1guilla")
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
         "--out_dir", args.out_dir],
        cwd=SCRIPT_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if proc.returncode != 0:
        print(f"  WARNING: plot_scaling.py exited with code {proc.returncode}")
else:
    # ── Inline scaling plot (fallback if plot_scaling.py is missing) ──
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
║    cost_breakdown_<NxMxK>.csv   – per-grid timing breakdown        ║
║    cost_perstep_<NxMxK>.csv     – raw per-step timings             ║
║    cost_scaling_stacked.pdf     – stacked bar comparison           ║
║    cost_scaling_loglog.pdf      – log-log scaling trend            ║
║    cost_scaling_pct.pdf         – percentage distribution          ║
╚══════════════════════════════════════════════════════════════════════╝
""")

sys.exit(0 if not failed_grids else 1)
