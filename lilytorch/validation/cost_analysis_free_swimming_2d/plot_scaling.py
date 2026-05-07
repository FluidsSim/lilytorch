#!/usr/bin/env python3
"""
Paper-quality multi-resolution scaling plot for the 2-D free-swimming cost analysis.

Reads CSV files produced by run_cost_analysis.py and generates:
  1. Stacked bar chart of grouped categories across grid sizes
  2. Log-log scaling plot showing how each category scales with N
  3. Normalised percentage bar chart

Usage
-----
    python plot_scaling.py                        # reads from ./figures/
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

parser = argparse.ArgumentParser(description="2-D multi-grid cost scaling plots")
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
# "3c " (trailing space) matches "3c   projection" but NOT the nested
# "3c.i Jacobi" / "3c.ii V-cycle" sub-timers (avoids double-counting).
CATEGORIES = {
    "Body update (SDF)":       ["1b"],
    "mu + normals":            ["2 "],
    "Convection & diffusion":  ["3a  "],
    "BDIM meta-equation":      ["3b"],
    "Projection (pressure)":   ["3c "],
    "set_BCs (2d)":            ["3d"],
    "var-density coeffs":      ["3e"],
    "release BDIM":            ["3f"],
    "Forces":                  ["4 "],
    "FARMS (apply_forces)":    ["6 "],
}
_OTHER_LABEL = "Other (residual)"

CAT_COLOURS = {
    "Body update (SDF)":       "#26a69a",
    "mu + normals":            "#66bb6a",
    "Convection & diffusion":  "#42a5f5",
    "BDIM meta-equation":      "#ab47bc",
    "Projection (pressure)":   "#ef5350",
    "set_BCs (2d)":            "#fbc02d",
    "var-density coeffs":      "#7cb342",
    "release BDIM":            "#bcaaa4",
    "Forces":                  "#ffa726",
    "FARMS (apply_forces)":    "#5c6bc0",
    _OTHER_LABEL:              "#90a4ae",
}

HATCHES = {
    "Body update (SDF)":       "",
    "mu + normals":            "",
    "Convection & diffusion":  "",
    "BDIM meta-equation":      "",
    "Projection (pressure)":   "",
    "set_BCs (2d)":            "",
    "var-density coeffs":      "",
    "release BDIM":            "",
    "Forces":                  "xx",
    "FARMS (apply_forces)":    "//",
    _OTHER_LABEL:              "..",
}


# ═══════════════════════════════════════════════════════════════════════
# Load CSVs  (2-D only: NxM, not NxMxK)
# ═══════════════════════════════════════════════════════════════════════

def _is_2d_csv(path):
    """True iff the filename encodes a 2-D grid (NxM, no third dimension)."""
    base = os.path.basename(path)
    return bool(re.search(r"cost_breakdown_\d+x\d+(?!x\d)", base))


def _grid_cells_from_path(p):
    m = re.search(r"(\d+)x(\d+)(?!x\d)", os.path.basename(p))
    return int(m.group(1)) * int(m.group(2)) if m else 0


csv_files = [f for f in glob.glob(os.path.join(args.data_dir, "cost_breakdown_*.csv"))
             if _is_2d_csv(f)]
if not csv_files:
    print(f"ERROR: No 2-D cost_breakdown_*.csv files found in {args.data_dir}")
    print("  Run: python run_cost_analysis.py --Nx <N> --Ny <M>")
    exit(1)

csv_files.sort(key=_grid_cells_from_path)

grids       = []
grid_labels = []
grid_cells  = []
cat_data    = {c: [] for c in CATEGORIES}
cat_data[_OTHER_LABEL] = []

for csv_f in csv_files:
    m = re.search(r"(\d+)x(\d+)(?!x\d)", os.path.basename(csv_f))
    if not m:
        continue
    nx, ny = int(m.group(1)), int(m.group(2))
    grids.append((nx, ny))
    grid_labels.append(f"{nx}×{ny}")
    grid_cells.append(nx * ny)

    df = pd.read_csv(csv_f)

    total_row   = df[df["component"] == "TOTAL step"]
    n_steps     = int(total_row["calls"].iloc[0]) if len(total_row) > 0 else 1
    total_step_s = float(total_row["total_s"].iloc[0]) if len(total_row) > 0 else 0.0

    # Prefer per-step medians (robust to recompile spikes)
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

    def _cat_ms_per_step(prefixes, _df=df, _df_ps=df_ps, _n_st=n_steps):
        if _df_ps is not None:
            matching_cols = [c for c in _df_ps.columns
                             if any(c.startswith(p) for p in prefixes)
                             and c != "TOTAL step"]
            if matching_cols:
                return float(_df_ps[matching_cols].sum(axis=1).median())
        mask = _df["component"].apply(
            lambda comp, pfx=prefixes: any(comp.startswith(p) for p in pfx))
        mask &= _df["component"] != "TOTAL step"
        return 1e3 * _df.loc[mask, "total_s"].sum() / _n_st

    def _total_ms_per_step(_df_ps=df_ps, _total_s=total_step_s, _n_st=n_steps):
        if _df_ps is not None and "TOTAL step" in _df_ps.columns:
            return float(_df_ps["TOTAL step"].median())
        return 1e3 * _total_s / _n_st

    explicit_ms = 0.0
    for cat_name, prefixes in CATEGORIES.items():
        cat_ms = _cat_ms_per_step(prefixes)
        explicit_ms += cat_ms
        cat_data[cat_name].append(cat_ms)

    total_ms = _total_ms_per_step()
    cat_data[_OTHER_LABEL].append(max(total_ms - explicit_ms, 0.0))

PLOT_ORDER = list(CATEGORIES.keys()) + [_OTHER_LABEL]

n_grids = len(grids)
if n_grids == 0:
    print("ERROR: No valid 2-D grid CSVs found.")
    exit(1)

print(f"  Loaded {n_grids} grid(s): {', '.join(grid_labels)}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Stacked bar chart
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * n_grids + 2), 4.0))
x         = np.arange(n_grids)
bar_width = 0.55
bottoms   = np.zeros(n_grids)

for cat_name in PLOT_ORDER:
    vals = np.array(cat_data[cat_name])
    if vals.sum() == 0:
        continue
    ax.bar(x, vals, bar_width, bottom=bottoms,
           label=cat_name,
           color=CAT_COLOURS.get(cat_name, "#bdbdbd"),
           hatch=HATCHES.get(cat_name, ""),
           edgecolor="white", linewidth=0.6)
    bottoms += vals

for i in range(n_grids):
    ax.text(x[i], bottoms[i] + 0.02 * bottoms.max(),
            f"{bottoms[i]:.1f} ms", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(grid_labels)
ax.set_xlabel("Grid resolution")
ax.set_ylabel("Time per step (ms)")
ax.set_title("Cost breakdown – 2-D pinned 1guilla")
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

    markers = ["o", "s", "D", "^", "v", "p", "h", "X", "*"]
    for i, cat_name in enumerate(PLOT_ORDER):
        vals = np.array(cat_data[cat_name])
        if vals.sum() == 0:
            continue
        ax2.loglog(cells_arr, vals,
                   marker=markers[i % len(markers)], markersize=5,
                   label=cat_name, linewidth=1.4,
                   color=CAT_COLOURS.get(cat_name, "#bdbdbd"))

    x_ref = np.array([cells_arr.min(), cells_arr.max()])
    proj_vals = cat_data.get("Projection (pressure)", [1.0])
    y0 = 0.3 * proj_vals[0] if proj_vals and proj_vals[0] > 0 else 1.0
    scale_linear = y0 * (x_ref / x_ref[0])
    scale_nlogn  = y0 * (x_ref / x_ref[0]) * np.log2(x_ref) / np.log2(x_ref[0])
    ax2.loglog(x_ref, scale_linear, "k--", alpha=0.35, linewidth=1.0,
               label=r"$\mathcal{O}(N)$")
    ax2.loglog(x_ref, scale_nlogn,  "k:",  alpha=0.35, linewidth=1.0,
               label=r"$\mathcal{O}(N\log N)$")

    # a + b·N fit on TOTAL step
    totals_arr = np.zeros_like(cells_arr)
    for cat_name in PLOT_ORDER:
        totals_arr = totals_arr + np.array(cat_data[cat_name])
    if np.all(totals_arr > 0) and len(cells_arr) >= 2:
        A = np.column_stack([np.ones_like(cells_arr), cells_arr])
        coef, *_ = np.linalg.lstsq(A, totals_arr, rcond=None)
        a_fit, b_fit = float(coef[0]), float(coef[1])
        if a_fit > 0 and b_fit > 0:
            x_dense = np.geomspace(x_ref[0], x_ref[1], 64)
            ax2.loglog(
                x_dense, a_fit + b_fit * x_dense,
                color="#37474f", linestyle=(0, (4, 2)), linewidth=1.2, alpha=0.7,
                label=fr"$a + b\,N$  ($a={a_fit:.2f}\,$ms, "
                      fr"$b={b_fit*1e6:.2f}\,$ns/cell)",
            )

    ax2.set_xlabel("Number of grid cells")
    ax2.set_ylabel("Time per step (ms)")
    ax2.set_title("Computational scaling – 2-D pinned 1guilla")
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
totals   = np.zeros(n_grids)
for cat_name in PLOT_ORDER:
    totals += np.array(cat_data[cat_name])

for cat_name in PLOT_ORDER:
    vals = np.array(cat_data[cat_name])
    if vals.sum() == 0:
        continue
    pcts = 100.0 * vals / np.maximum(totals, 1e-12)
    ax3.bar(x, pcts, bar_width, bottom=bottoms3,
            label=cat_name,
            color=CAT_COLOURS.get(cat_name, "#bdbdbd"),
            hatch=HATCHES.get(cat_name, ""),
            edgecolor="white", linewidth=0.6)
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
ax3.set_title("Relative cost distribution – 2-D pinned 1guilla")
ax3.set_ylim(0, 105)
ax3.legend(loc="upper right", framealpha=0.9, fontsize=7.5, ncol=1)
fig3.tight_layout()

pct_path = os.path.join(args.out_dir, "cost_scaling_pct.pdf")
fig3.savefig(pct_path)
fig3.savefig(pct_path.replace(".pdf", ".png"))
print(f"  Figure saved → {pct_path}")
plt.close(fig3)

print("\n  Done. All figures saved to:", args.out_dir)
