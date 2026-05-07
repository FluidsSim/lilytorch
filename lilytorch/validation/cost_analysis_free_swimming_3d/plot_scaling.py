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

parser = argparse.ArgumentParser(description="Multi-grid cost sc" \
"aling plots")
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
# The explicit categories map to *leaf* timers instrumented in
# ``run_cost_analysis.py``.  "Other (residual)" is computed per grid as
#     residual = TOTAL step  −  Σ (explicit leaves)
# so that every bit of step cost (parent-timer overhead, bookkeeping,
# set_BCs, etc.) is represented.  Smagorinsky is deliberately excluded
# because the cost run sets ``smagorinsky_cs = 0``.
# Each prefix matches ONLY the outer leaf timer for its category.  In
# particular, ``"3c "`` (trailing space) matches
# ``"3c   projection (Poisson+gradient+correction)"`` but NOT
# ``"3c.i Jacobi smoothing"`` or ``"3c.ii V-cycle (top-level)"`` — the
# latter two are nested sub-timers *inside* projection and would
# double-count it on grids where Poisson internals are instrumented
# (≥ 500k cells).
CATEGORIES = {
    "Body update (SDF)":       ["1b"],
    "mu + normals":            ["2 "],
    "Convection & diffusion":  ["3a  "],
    "BDIM meta-equation":      ["3b"],
    "Projection (pressure)":   ["3c "],
    "set_BCs (3d)":            ["3d"],
    "var-density coeffs (3e)": ["3e"],
    "release BDIM (3f)":       ["3f"],
    "Forces":                  ["4 "],
    "Plotting & saving":       ["5 "],
    "FARMS (apply_forces)":    ["6 "],
}
_OTHER_LABEL = "Other (residual)"

CAT_COLOURS = {
    "Body update (SDF)":       "#26a69a",
    "mu + normals":            "#66bb6a",
    "Convection & diffusion":  "#42a5f5",
    "BDIM meta-equation":      "#ab47bc",   # violet: distinct from adv/diff blue
    "Projection (pressure)":   "#ef5350",
    "set_BCs (3d)":            "#fbc02d",
    "var-density coeffs (3e)": "#7cb342",
    "release BDIM (3f)":       "#bcaaa4",
    "Forces":                  "#ffa726",
    "Plotting & saving":       "#8d6e63",
    "FARMS (apply_forces)":    "#5c6bc0",
    _OTHER_LABEL:              "#90a4ae",
}

HATCHES = {
    "Body update (SDF)":       "",
    "mu + normals":            "",
    "Convection & diffusion":  "",
    "BDIM meta-equation":      "",
    "Projection (pressure)":   "",
    "set_BCs (3d)":            "",
    "var-density coeffs (3e)": "",
    "release BDIM (3f)":       "",
    "Forces":                  "xx",
    "Plotting & saving":       "",
    "FARMS (apply_forces)":    "//",
    _OTHER_LABEL:              "..",
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
cat_data[_OTHER_LABEL] = []              # residual

for csv_f in csv_files:
    m = re.search(r"(\d+)x(\d+)x(\d+)", os.path.basename(csv_f))
    if not m:
        continue
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    grids.append((nx, ny, nz))
    grid_labels.append(f"{nx}×{ny}×{nz}")
    grid_cells.append(nx * ny * nz)

    df = pd.read_csv(csv_f)

    # Number of measured steps = calls of "TOTAL step"
    total_row = df[df["component"] == "TOTAL step"]
    n_steps = int(total_row["calls"].iloc[0]) if len(total_row) > 0 else 1
    total_step_s = (
        float(total_row["total_s"].iloc[0]) if len(total_row) > 0 else 0.0
    )

    # Prefer per-step medians when a matching cost_perstep_*.csv is
    # available.  Per-step medians are robust to the single-recompile
    # outliers that can inflate dynamic-shape compiled kernel means by
    # orders of magnitude (e.g. one 225 ms spike across 50 steps pushes
    # mu+normals mean from 0.5 ms to 5 ms).
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
                per_step_sum = _df_ps[matching_cols].sum(axis=1)
                return float(per_step_sum.median())
        mask = _df["component"].apply(
            lambda comp, pfx=prefixes: any(comp.startswith(p) for p in pfx))
        mask &= _df["component"] != "TOTAL step"
        return 1e3 * _df.loc[mask, "total_s"].sum() / _n_st

    def _total_ms_per_step(_df_ps=df_ps, _total_s=total_step_s, _n_st=n_steps):
        if _df_ps is not None and "TOTAL step" in _df_ps.columns:
            return float(_df_ps["TOTAL step"].median())
        return 1e3 * _total_s / _n_st

    # Explicit leaf categories
    explicit_ms = 0.0
    for cat_name, prefixes in CATEGORIES.items():
        cat_ms = _cat_ms_per_step(prefixes)
        explicit_ms += cat_ms
        cat_data[cat_name].append(cat_ms)

    # Residual — everything in TOTAL step not claimed above.
    total_ms = _total_ms_per_step()
    residual_ms = max(total_ms - explicit_ms, 0.0)
    cat_data[_OTHER_LABEL].append(residual_ms)

# Ordered list of categories used for plotting (explicit first, residual last).
PLOT_ORDER = list(CATEGORIES.keys()) + [_OTHER_LABEL]

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

    markers = ["o", "s", "D", "^", "v", "p", "h", "X", "*"]
    for i, cat_name in enumerate(PLOT_ORDER):
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

    # ── a + b·N reference fitted to TOTAL step ──────────────────────
    # The plateau-investigation plan asks for an explicit ``a + b·N``
    # overlay on the per-condition log-log so the constant-cost floor
    # is visually obvious.  ``a`` (intercept) is the per-step launch /
    # FARMS / Python-overhead floor; ``b`` is the asymptotic per-cell
    # slope.  We fit a non-negative least-squares to the TOTAL-step
    # series (residual + every category) so the curve always exists,
    # even when the smallest grid is launch-overhead-bound.
    totals_arr = np.zeros_like(cells_arr)
    for cat_name in PLOT_ORDER:
        totals_arr = totals_arr + np.array(cat_data[cat_name])
    if np.all(totals_arr > 0) and len(cells_arr) >= 2:
        # Linear least-squares on (1, N) — closed form, no clipping
        # needed because TOTAL is strictly positive.
        A = np.column_stack([np.ones_like(cells_arr), cells_arr])
        coef, *_ = np.linalg.lstsq(A, totals_arr, rcond=None)
        a_fit, b_fit = float(coef[0]), float(coef[1])
        if a_fit > 0 and b_fit > 0:
            x_dense = np.geomspace(x_ref[0], x_ref[1], 64)
            ax2.loglog(
                x_dense, a_fit + b_fit * x_dense,
                color="#37474f", linestyle=(0, (4, 2)), linewidth=1.2,
                alpha=0.7,
                label=fr"$a + b\,N$  ($a={a_fit:.2f}\,$ms, "
                      fr"$b={b_fit*1e6:.2f}\,$ns/cell)",
            )

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
