#!/usr/bin/env python3
"""
Paper-quality plots for the body-fraction cost analysis.

Reads CSVs produced by ``run_body_fraction_analysis.py`` and generates
per-domain log-log cost curves comparing ``python`` vs ``kernel``, plus
a stacked-bar breakdown and a kernel speed-up summary.

Expected filename pattern:
    cost_breakdown_{Nx}x{Ny}x{Nz}_{domain}_{mode}.csv
where ``mode ∈ {python, kernel}`` and ``domain`` is one of the preset
names in ``run_body_fraction_analysis.py``.

Usage
-----
    python plot_body_fraction.py                       # reads ./figures/body_fraction/
    python plot_body_fraction.py --data_dir <path>
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

parser = argparse.ArgumentParser(description="Body-fraction cost plots")
parser.add_argument("--data_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "figures", "body_fraction"))
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()
args.data_dir = os.path.abspath(args.data_dir)
if args.out_dir is None:
    args.out_dir = args.data_dir
else:
    args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Publication style (matches plot_scaling.py)
# ─────────────────────────────────────────────────────────────────────
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

# Paper categories (same taxonomy as plot_scaling.py).
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
PLOT_ORDER = list(CATEGORIES.keys()) + [_OTHER_LABEL]

CAT_COLOURS = {
    "Body update (SDF)":       "#26a69a",
    "mu + normals":            "#66bb6a",
    "Convection & diffusion":  "#42a5f5",
    "BDIM meta-equation":      "#ab47bc",
    "Projection (pressure)":   "#ef5350",
    "set_BCs (3d)":            "#fbc02d",
    "var-density coeffs (3e)": "#7cb342",
    "release BDIM (3f)":       "#bcaaa4",
    "Forces":                  "#ffa726",
    "Plotting & saving":       "#8d6e63",
    "FARMS (apply_forces)":    "#5c6bc0",
    _OTHER_LABEL:              "#90a4ae",
}

MODE_STYLES = {
    "python": {"linestyle": "--", "marker": "s",
                "label": "python mode"},
    "kernel": {"linestyle": "-",  "marker": "o",
                "label": "kernel mode"},
}
MODE_ALIASES = {
    "nboff": "python",
    "nbon": "kernel",
    "nbforces_opt": "kernel",
}
MODE_PLOT_ORDER = ["python", "kernel"]

DOMAIN_LABEL = {
    "small": "Small domain (Lx = 1.2 m,  body ≈ 70 % of domain)",
    "large": "Large domain (Lx = 2.7 m,  body ≈ 30 % of domain)",
}

# Sort domains so plots and legends always read small → large.
DOMAIN_ORDER = ["small", "large"]


# ─────────────────────────────────────────────────────────────────────
# Parse CSV filenames
# ─────────────────────────────────────────────────────────────────────
CSV_RE = re.compile(
    r"cost_breakdown_(\d+)x(\d+)x(\d+)_([A-Za-z]+)_(python|kernel|nboff|nbon|nbforces_opt)\.csv$")

csv_files = glob.glob(os.path.join(args.data_dir, "cost_breakdown_*.csv"))
if not csv_files:
    print(f"ERROR: No cost_breakdown_*.csv files found in {args.data_dir}")
    raise SystemExit(1)

# Organise: records[domain][mode] -> list of (nx, ny, nz, cell_counts, cat_ms)
records = {}

for csv_f in csv_files:
    m = CSV_RE.search(os.path.basename(csv_f))
    if not m:
        continue
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    domain = m.group(4)
    nb     = MODE_ALIASES.get(m.group(5), m.group(5))

    df = pd.read_csv(csv_f)
    total_row = df[df["component"] == "TOTAL step"]
    if len(total_row) == 0:
        print(f"  skip {os.path.basename(csv_f)}: no TOTAL step row")
        continue
    n_steps = int(total_row["calls"].iloc[0])
    total_step_s = float(total_row["total_s"].iloc[0])

    # Per-step medians (preferred, robust to recompile outliers)
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
            cols = [c for c in _df_ps.columns
                    if any(c.startswith(p) for p in prefixes)
                    and c != "TOTAL step"]
            if cols:
                return float(_df_ps[cols].sum(axis=1).median())
        mask = _df["component"].apply(
            lambda c, pfx=prefixes: any(c.startswith(p) for p in pfx))
        mask &= _df["component"] != "TOTAL step"
        return 1e3 * _df.loc[mask, "total_s"].sum() / _n_st

    def _total_ms_per_step():
        if df_ps is not None and "TOTAL step" in df_ps.columns:
            return float(df_ps["TOTAL step"].median())
        return 1e3 * total_step_s / n_steps

    cat_ms = {}
    explicit = 0.0
    for cat_name, prefixes in CATEGORIES.items():
        v = _cat_ms_per_step(prefixes)
        cat_ms[cat_name] = v
        explicit += v
    total_ms = _total_ms_per_step()
    cat_ms[_OTHER_LABEL] = max(total_ms - explicit, 0.0)
    cat_ms["TOTAL step"] = total_ms

    records.setdefault(domain, {}).setdefault(nb, []).append({
        "grid":    (nx, ny, nz),
        "cells":   nx * ny * nz,
        "cat_ms":  cat_ms,
    })

if not records:
    print("ERROR: No recognisable CSVs (expected "
        "cost_breakdown_{Nx}x{Ny}x{Nz}_{domain}_{python|kernel}.csv)")
    raise SystemExit(1)

# Sort each list by cell count
for d in records:
    for nb in records[d]:
        records[d][nb].sort(key=lambda r: r["cells"])

# Iterate in canonical small → large order (then any extra domains by name)
def _domain_sort_key(name):
    try:
        return (0, DOMAIN_ORDER.index(name))
    except ValueError:
        return (1, name)
records = {d: records[d] for d in sorted(records, key=_domain_sort_key)}

print(f"  Loaded {sum(len(v) for d in records.values() for v in d.values())} "
      f"CSV(s) across {len(records)} domain(s)")


# ─────────────────────────────────────────────────────────────────────
# Figure 1 — per-domain log-log scaling, python vs kernel
# ─────────────────────────────────────────────────────────────────────
for domain, by_nb in records.items():
    if not by_nb:
        continue
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for nb, runs in by_nb.items():
        if not runs:
            continue
        cells = np.array([r["cells"] for r in runs], dtype=float)
        total = np.array([r["cat_ms"]["TOTAL step"] for r in runs])
        style = MODE_STYLES[nb]
        ax.loglog(cells, total,
                  linestyle=style["linestyle"],
                  marker=style["marker"], markersize=6,
                  linewidth=1.6, color="#263238",
                  label=f"TOTAL  ({style['label']})")
        for cat_name in PLOT_ORDER:
            vals = np.array([r["cat_ms"][cat_name] for r in runs])
            if vals.sum() == 0:
                continue
            ax.loglog(cells, vals,
                      linestyle=style["linestyle"],
                      marker=style["marker"], markersize=4,
                      linewidth=1.0, alpha=0.9,
                      color=CAT_COLOURS.get(cat_name, "#bdbdbd"),
                      label=f"{cat_name}  ({style['label']})"
                            if nb == "kernel" else None)

    # Reference slope lines (ideal O(N) at arbitrary y-anchor)
    all_cells = sorted({r["cells"]
                        for runs in by_nb.values() for r in runs})
    if len(all_cells) >= 2:
        x_ref = np.array([all_cells[0], all_cells[-1]], dtype=float)
        # Anchor at TOTAL of smallest grid (python if available else first mode)
        base_nb = "python" if "python" in by_nb and by_nb["python"] else next(iter(by_nb))
        y_anchor = by_nb[base_nb][0]["cat_ms"]["TOTAL step"]
        y_ref = y_anchor * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e",
                  linestyle=":", linewidth=1.0, alpha=0.7,
                  label="O(N) reference")

        # ── a + b·N reference fitted to the kernel mode ─
        # Surfaces the constant-cost floor of the optimised path.
        fit_nb = None
        for cand in ("kernel", "python"):
            if cand in by_nb and len(by_nb[cand]) >= 2:
                fit_nb = cand
                break
        if fit_nb is not None:
            runs = by_nb[fit_nb]
            cells_fit = np.array([r["cells"] for r in runs], dtype=float)
            total_fit = np.array([r["cat_ms"]["TOTAL step"] for r in runs])
            A = np.column_stack([np.ones_like(cells_fit), cells_fit])
            coef, *_ = np.linalg.lstsq(A, total_fit, rcond=None)
            a_fit, b_fit = float(coef[0]), float(coef[1])
            if a_fit > 0 and b_fit > 0:
                x_dense = np.geomspace(x_ref[0], x_ref[1], 64)
                ax.loglog(
                    x_dense, a_fit + b_fit * x_dense,
                    color="#37474f", linestyle=(0, (4, 2)),
                    linewidth=1.2, alpha=0.7,
                    label=fr"$a + b\,N$ fit ({fit_nb}; "
                          fr"$a={a_fit:.2f}$ ms, "
                          fr"$b={b_fit*1e6:.2f}$ ns/cell)",
                )

    ax.set_xlabel("Total cells  $N_x N_y N_z$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title(DOMAIN_LABEL.get(domain, f"Domain: {domain}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.9, ncol=1, fontsize=7.5)
    fig.tight_layout()
    path = os.path.join(args.out_dir, f"body_fraction_loglog_{domain}.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Figure saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 2 — stacked bars per domain comparing kernel vs python
# ─────────────────────────────────────────────────────────────────────
for domain, by_nb in records.items():
    if len(by_nb) < 1:
        continue
    # Build cell-count axis from the union of python/kernel grids.
    all_grids = sorted({tuple(r["grid"])
                        for runs in by_nb.values() for r in runs},
                       key=lambda g: g[0] * g[1] * g[2])
    n_g = len(all_grids)
    if n_g == 0:
        continue

    nb_order = [nb for nb in ("python", "kernel") if nb in by_nb]
    n_nb = len(nb_order)

    fig, ax = plt.subplots(
        figsize=(max(5.5, 1.4 * n_g * n_nb + 2), 4.4))
    bar_w = 0.38
    group_x = np.arange(n_g)

    for j, nb in enumerate(nb_order):
        runs = {tuple(r["grid"]): r for r in by_nb[nb]}
        offsets = (j - (n_nb - 1) / 2) * bar_w
        xs = group_x + offsets
        bottoms = np.zeros(n_g)
        for cat_name in PLOT_ORDER:
            vals = np.array([runs[g]["cat_ms"].get(cat_name, 0.0)
                             if g in runs else 0.0
                             for g in all_grids])
            if vals.sum() == 0:
                continue
            ax.bar(xs, vals, bar_w, bottom=bottoms,
                   color=CAT_COLOURS.get(cat_name, "#bdbdbd"),
                   edgecolor="white", linewidth=0.4,
                   label=cat_name if j == 0 else None,
                   hatch="" if nb == "kernel" else "//")
            bottoms += vals
        # Totals
        for xi, b in zip(xs, bottoms):
            ax.text(xi, b * 1.02, f"{b:.1f}",
                    ha="center", va="bottom", fontsize=7)
        # nb label under each group column
        for xi, g in zip(xs, all_grids):
            ax.text(xi, -0.02 * bottoms.max(), MODE_STYLES[nb]["label"].split()[0],
                    ha="center", va="top", fontsize=6.5, color="#455a64")

    ax.set_xticks(group_x)
    ax.set_xticklabels([f"{g[0]}×{g[1]}×{g[2]}" for g in all_grids])
    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title(DOMAIN_LABEL.get(domain, f"Domain: {domain}")
                 + "  —  hatched = python baseline")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.9, fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(args.out_dir, f"body_fraction_stacked_{domain}.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Figure saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 3 — combined 4-curve log-log view (all domain × mode curves)
# ─────────────────────────────────────────────────────────────────────
# Headline plot for the body-fraction comparison: every (domain, mode)
# combination on the same log-log axes so the kernel benefit
# emerges directly from the curve spacing.
#
# Expected pattern at convergence:
#   * "small" domain: kernel and python curves nearly overlap (body fills
#     ~ 70 % of cells, so the optimised path has less work to skip).
#   * "large" domain: kernel sits well below python, with the gap widening
#     as N grows.
DOMAIN_COND_COLOURS = {
    "small": "#1976d2",   # blue
    "large": "#d32f2f",   # red
}

if records:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _MARKER_FACE = {"python": "filled", "kernel": "white"}
    for domain, by_nb in records.items():
        col = DOMAIN_COND_COLOURS.get(domain, "#455a64")
        for nb in MODE_PLOT_ORDER:
            runs = by_nb.get(nb, [])
            if not runs:
                continue
            cells = np.array([r["cells"] for r in runs], dtype=float)
            total = np.array([r["cat_ms"]["TOTAL step"] for r in runs])
            style = MODE_STYLES[nb]
            label = (f"{DOMAIN_LABEL.get(domain, domain).split('(')[0].strip()} "
                     f"— {style['label']}")
            face = "white" if _MARKER_FACE.get(nb) == "white" else col
            ax.loglog(cells, total,
                      linestyle=style["linestyle"],
                      marker=style["marker"], markersize=6,
                      linewidth=1.8, color=col,
                      markerfacecolor=face,
                      markeredgewidth=1.4,
                      label=label)

    # O(N) reference slope, anchored at the smallest data point overall.
    all_pts = [(r["cells"], r["cat_ms"]["TOTAL step"])
               for by_nb in records.values()
               for runs in by_nb.values() for r in runs]
    if len(all_pts) >= 2:
        all_pts.sort()
        x_ref = np.array([all_pts[0][0], all_pts[-1][0]], dtype=float)
        y_anchor = all_pts[0][1]
        y_ref = y_anchor * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e",
                  linestyle=":", linewidth=1.0, alpha=0.7,
                  label="O(N) reference")

    ax.set_xlabel("Total cells  $N_x N_y N_z$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Cost scaling: domain × solver mode")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "body_fraction_loglog_combined.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Figure saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 4 — speed-ups vs python baseline, per domain, vs N.
# ─────────────────────────────────────────────────────────────────────
# Plots TIME(python) / TIME(kernel) so the body-fraction-dependent win of
# the optimised path is visible at a glance.
if any(any(nb != "python" for nb in by_nb) and "python" in by_nb
       for by_nb in records.values()):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for domain, by_nb in records.items():
        if "python" not in by_nb:
            continue
        col = DOMAIN_COND_COLOURS.get(domain, "#455a64")
        off = {tuple(r["grid"]): r for r in by_nb["python"]}
        for nb in MODE_PLOT_ORDER:
            if nb == "python" or nb not in by_nb:
                continue
            other = {tuple(r["grid"]): r for r in by_nb[nb]}
            common = sorted(set(off) & set(other),
                            key=lambda g: g[0] * g[1] * g[2])
            if not common:
                continue
            cells = np.array([g[0] * g[1] * g[2] for g in common],
                             dtype=float)
            speedup = np.array([
                off[g]["cat_ms"]["TOTAL step"]
                / other[g]["cat_ms"]["TOTAL step"]
                for g in common
            ])
            style = MODE_STYLES[nb]
            dom_short = DOMAIN_LABEL.get(domain, domain).split("(")[0].strip()
            ax.semilogx(cells, speedup,
                        linestyle=style["linestyle"],
                        marker=style["marker"], markersize=6,
                        linewidth=1.6, color=col,
                        label=f"{dom_short} — {style['label']}")

    ax.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=1.0,
               label="break-even (1×)")
    ax.set_xlabel("Total cells  $N_x N_y N_z$")
    ax.set_ylabel(r"Speed-up  $T_\mathrm{python}\,/\,T_\mathrm{mode}$")
    ax.set_title("Kernel speed-up vs python baseline")
    ax.legend(loc="best", framealpha=0.9, fontsize=7.5)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "body_fraction_speedup.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Figure saved → {path}")
    plt.close(fig)


print(f"\n  Done. Figures saved to: {args.out_dir}")
