#!/usr/bin/env python3
"""
Paper-quality multi-resolution scaling plot for the free-swimming cost analysis.

Reads CSV files produced by run_cost_analysis.py and generates:
  1. Stacked bar chart of grouped categories across grid sizes
  2. Log-log scaling plot showing how each category scales with N

Usage
-----
    python plot_scaling.py                       # reads from ./figures/
    python plot_scaling.py --data_dir ./figures/  # explicit path
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Multi-grid cost scaling plots")
parser.add_argument("--data_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "figures"),
                    help="Directory containing cost_breakdown_*.csv files")
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()
if args.out_dir is None:
    args.out_dir = args.data_dir

# ── Publication style ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# ── Paper categories (must match run_cost_analysis.py prefixes) ──────
CATEGORIES = {
    "Body update (SDF)":       ["1b"],
    "Forces":                  ["5 "],
    "Projection (pressure)":   ["4d", "4e ", "4f", "4g", "4h"],
    "Convection & diffusion":  ["4a", "4b"],
    "Other":                   ["1a", "7 ", "2 ", "3 ", "4c", "6 "],
}

CAT_COLOURS = {
    "Body update (SDF)":       "#26a69a",
    "Forces":                  "#ffa726",
    "Projection (pressure)":   "#ef5350",
    "Convection & diffusion":  "#42a5f5",
    "Other":                   "#90a4ae",
}

HATCHES = {
    "Body update (SDF)":       "",
    "Forces":                  "xx",
    "Projection (pressure)":   "",
    "Convection & diffusion":  "",
    "Other":                   "..",
}


# ═══════════════════════════════════════════════════════════════════════
# Load CSVs
# ═══════════════════════════════════════════════════════════════════════

csv_files = glob.glob(os.path.join(args.data_dir, "cost_breakdown_*.csv"))
if not csv_files:
    print(f"ERROR: No cost_breakdown_*.csv files found in {args.data_dir}")
    print("  Run: python run_cost_analysis.py --Nx <N> --Ny <M> --Nz <K>")
    exit(1)

# Sort by total cell count (not alphabetically)
def _grid_cells_from_path(p):
    m = re.search(r"(\d+)x(\d+)x(\d+)", os.path.basename(p))
    return int(m.group(1)) * int(m.group(2)) * int(m.group(3)) if m else 0
csv_files.sort(key=_grid_cells_from_path)

grids = []       # list of (Nx, Ny, Nz)
grid_labels = [] # "128×32×32"
grid_cells = []  # total cells
cat_data = {c: [] for c in CATEGORIES}  # cat → list of mean_ms per grid

for csv_f in csv_files:
    m = re.search(r"(\d+)x(\d+)x(\d+)", os.path.basename(csv_f))
    if not m:
        continue
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    grids.append((nx, ny, nz))
    grid_labels.append(f"{nx}×{ny}×{nz}")
    grid_cells.append(nx * ny * nz)

    df = pd.read_csv(csv_f)

    # Compute per-category mean time
    for cat_name, prefixes in CATEGORIES.items():
        mask = df["component"].apply(
            lambda comp: any(comp.startswith(pfx) for pfx in prefixes))
        cat_total_s = df.loc[mask, "total_s"].sum()
        # Number of measured steps = calls of "TOTAL step"
        total_row = df[df["component"] == "TOTAL step"]
        n_steps = int(total_row["calls"].iloc[0]) if len(total_row) > 0 else 1
        cat_mean_ms = 1e3 * cat_total_s / n_steps
        cat_data[cat_name].append(cat_mean_ms)

n_grids = len(grids)
if n_grids == 0:
    print("ERROR: No valid grid CSVs found.")
    exit(1)

print(f"  Loaded {n_grids} grid(s): {', '.join(grid_labels)}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Stacked bar chart
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n_grids + 2), 4.0))
x = np.arange(n_grids)
bar_width = 0.55
bottoms = np.zeros(n_grids)

for cat_name in CATEGORIES:
    vals = np.array(cat_data[cat_name])
    if vals.sum() == 0:
        continue
    ax.bar(x, vals, bar_width, bottom=bottoms,
           label=cat_name,
           color=CAT_COLOURS.get(cat_name, "#bdbdbd"),
           hatch=HATCHES.get(cat_name, ""),
           edgecolor="white", linewidth=0.6)
    bottoms += vals

# Total step time annotation
for i in range(n_grids):
    ax.text(x[i], bottoms[i] + 0.02 * bottoms.max(),
            f"{bottoms[i]:.1f} ms", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(grid_labels)
ax.set_xlabel("Grid resolution")
ax.set_ylabel("Time per step (ms)")
ax.set_title("Cost breakdown – free-swimming 1guilla")
ax.legend(loc="upper left", framealpha=0.9, ncol=1)
ax.set_ylim(0, bottoms.max() * 1.15)
fig.tight_layout()

stacked_path = os.path.join(args.out_dir, "cost_scaling_stacked.pdf")
fig.savefig(stacked_path)
fig.savefig(stacked_path.replace(".pdf", ".png"))
print(f"  Figure saved → {stacked_path}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Log–log scaling
# ═══════════════════════════════════════════════════════════════════════

if n_grids >= 2:
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
    cells_arr = np.array(grid_cells, dtype=float)

    markers = ["o", "s", "D", "^", "v", "p", "h", "X"]
    for i, cat_name in enumerate(CATEGORIES):
        vals = np.array(cat_data[cat_name])
        if vals.sum() == 0:
            continue
        ax2.loglog(cells_arr, vals,
                   marker=markers[i % len(markers)], markersize=5,
                   label=cat_name, linewidth=1.4,
                   color=CAT_COLOURS.get(cat_name, "#bdbdbd"))

    # Reference lines
    x_ref = np.array([cells_arr.min(), cells_arr.max()])
    y0 = 0.3 * cat_data["Projection (pressure)"][0] if cat_data["Projection (pressure)"][0] > 0 else 1.0
    scale_linear = y0 * (x_ref / x_ref[0])
    scale_nlogn  = y0 * (x_ref / x_ref[0]) * np.log2(x_ref) / np.log2(x_ref[0])
    ax2.loglog(x_ref, scale_linear, "k--", alpha=0.35, linewidth=1.0, label=r"$\mathcal{O}(N)$")
    ax2.loglog(x_ref, scale_nlogn,  "k:",  alpha=0.35, linewidth=1.0, label=r"$\mathcal{O}(N\log N)$")

    ax2.set_xlabel("Number of grid cells")
    ax2.set_ylabel("Time per step (ms)")
    ax2.set_title("Computational scaling – free-swimming 1guilla")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7.5, ncol=2)
    fig2.tight_layout()

    scaling_path = os.path.join(args.out_dir, "cost_scaling_loglog.pdf")
    fig2.savefig(scaling_path)
    fig2.savefig(scaling_path.replace(".pdf", ".png"))
    print(f"  Figure saved → {scaling_path}")
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Percentage stacked bar (normalised)
# ═══════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(max(4.5, 1.2 * n_grids + 2), 4.0))
bottoms3 = np.zeros(n_grids)
totals = np.zeros(n_grids)
for cat_name in CATEGORIES:
    totals += np.array(cat_data[cat_name])

for cat_name in CATEGORIES:
    vals = np.array(cat_data[cat_name])
    if vals.sum() == 0:
        continue
    pcts = 100.0 * vals / np.maximum(totals, 1e-12)
    ax3.bar(x, pcts, bar_width, bottom=bottoms3,
            label=cat_name,
            color=CAT_COLOURS.get(cat_name, "#bdbdbd"),
            hatch=HATCHES.get(cat_name, ""),
            edgecolor="white", linewidth=0.6)
    # Label segments ≥ 8%
    for j in range(n_grids):
        if pcts[j] >= 8:
            ax3.text(x[j], bottoms3[j] + pcts[j] / 2,
                     f"{pcts[j]:.0f}%", ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")
    bottoms3 += pcts

ax3.set_xticks(x)
ax3.set_xticklabels(grid_labels)
ax3.set_xlabel("Grid resolution")
ax3.set_ylabel("Fraction of step time (%)")
ax3.set_title("Relative cost distribution – free-swimming 1guilla")
ax3.set_ylim(0, 105)
ax3.legend(loc="upper right", framealpha=0.9, fontsize=7.5, ncol=1)
fig3.tight_layout()

pct_path = os.path.join(args.out_dir, "cost_scaling_pct.pdf")
fig3.savefig(pct_path)
fig3.savefig(pct_path.replace(".pdf", ".png"))
print(f"  Figure saved → {pct_path}")
plt.close(fig3)

print("\n  Done. All figures saved to:", args.out_dir)
