#!/usr/bin/env python3
"""
Paper-quality plots for the body-fraction cost analysis.

Reads CSVs produced by ``run_body_fraction_analysis.py`` and generates
per-domain log-log cost curves comparing narrow-band on vs off, plus
a stacked-bar breakdown and a narrow-band speed-up summary.

Expected filename pattern:
    cost_breakdown_{Nx}x{Ny}x{Nz}_{domain}_{nb}.csv
where ``nb ∈ {nbon, nboff}`` and ``domain`` is one of the preset names
in ``run_body_fraction_analysis.py``.

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
if args.out_dir is None:
    args.out_dir = args.data_dir
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
    "Forces":                  ["4 "],
}
_OTHER_LABEL = "Other (residual)"
PLOT_ORDER = list(CATEGORIES.keys()) + [_OTHER_LABEL]

CAT_COLOURS = {
    "Body update (SDF)":       "#26a69a",
    "mu + normals":            "#66bb6a",
    "Convection & diffusion":  "#42a5f5",
    "BDIM meta-equation":      "#ab47bc",
    "Projection (pressure)":   "#ef5350",
    "Forces":                  "#ffa726",
    _OTHER_LABEL:              "#90a4ae",
}

NB_STYLES = {
    "nbon":  {"linestyle": "-",  "marker": "o", "label": "narrow-band on"},
    "nboff": {"linestyle": "--", "marker": "s", "label": "full-grid (baseline)"},
}

DOMAIN_LABEL = {
    "small": "Small domain (Lx = 1.0 m,  body fraction ≈ 19 %)",
    "large": "Large domain (Lx = 2.4 m,  body fraction ≈ 1.4 %)",
}


# ─────────────────────────────────────────────────────────────────────
# Parse CSV filenames
# ─────────────────────────────────────────────────────────────────────
CSV_RE = re.compile(
    r"cost_breakdown_(\d+)x(\d+)x(\d+)_([A-Za-z]+)_(nbon|nboff)\.csv$")

csv_files = glob.glob(os.path.join(args.data_dir, "cost_breakdown_*.csv"))
if not csv_files:
    print(f"ERROR: No cost_breakdown_*.csv files found in {args.data_dir}")
    raise SystemExit(1)

# Organise: records[domain][nb] -> list of (nx, ny, nz, cell_counts, cat_ms)
records = {}

for csv_f in csv_files:
    m = CSV_RE.search(os.path.basename(csv_f))
    if not m:
        continue
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
    domain = m.group(4)
    nb     = m.group(5)

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
          "cost_breakdown_{Nx}x{Ny}x{Nz}_{domain}_{nbon|nboff}.csv)")
    raise SystemExit(1)

# Sort each list by cell count
for d in records:
    for nb in records[d]:
        records[d][nb].sort(key=lambda r: r["cells"])

print(f"  Loaded {sum(len(v) for d in records.values() for v in d.values())} "
      f"CSV(s) across {len(records)} domain(s)")


# ─────────────────────────────────────────────────────────────────────
# Figure 1 — per-domain log-log scaling, NB on vs NB off
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
        style = NB_STYLES[nb]
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
                            if nb == "nbon" else None)

    # Reference slope lines (ideal O(N) at arbitrary y-anchor)
    all_cells = sorted({r["cells"]
                        for runs in by_nb.values() for r in runs})
    if len(all_cells) >= 2:
        x_ref = np.array([all_cells[0], all_cells[-1]], dtype=float)
        # Anchor at TOTAL of smallest grid (NB-off if available else first nb)
        base_nb = "nboff" if "nboff" in by_nb and by_nb["nboff"] else next(iter(by_nb))
        y_anchor = by_nb[base_nb][0]["cat_ms"]["TOTAL step"]
        y_ref = y_anchor * x_ref / x_ref[0]
        ax.loglog(x_ref, y_ref, color="#9e9e9e",
                  linestyle=":", linewidth=1.0, alpha=0.7,
                  label="O(N) reference")

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
# Figure 2 — stacked bars per domain comparing nbon vs nboff
# ─────────────────────────────────────────────────────────────────────
for domain, by_nb in records.items():
    if len(by_nb) < 1:
        continue
    # Build cell-count axis from the union of nbon/nboff grids.
    all_grids = sorted({tuple(r["grid"])
                        for runs in by_nb.values() for r in runs},
                       key=lambda g: g[0] * g[1] * g[2])
    n_g = len(all_grids)
    if n_g == 0:
        continue

    nb_order = [nb for nb in ("nboff", "nbon") if nb in by_nb]
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
                   hatch="" if nb == "nbon" else "//")
            bottoms += vals
        # Totals
        for xi, b in zip(xs, bottoms):
            ax.text(xi, b * 1.02, f"{b:.1f}",
                    ha="center", va="bottom", fontsize=7)
        # nb label under each group column
        for xi, g in zip(xs, all_grids):
            ax.text(xi, -0.02 * bottoms.max(), NB_STYLES[nb]["label"].split()[0],
                    ha="center", va="top", fontsize=6.5, color="#455a64")

    ax.set_xticks(group_x)
    ax.set_xticklabels([f"{g[0]}×{g[1]}×{g[2]}" for g in all_grids])
    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title(DOMAIN_LABEL.get(domain, f"Domain: {domain}")
                 + "  —  hatched = full-grid baseline")
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
# Figure 3 — narrow-band speed-up (nboff / nbon) vs N, per domain
# ─────────────────────────────────────────────────────────────────────
if any("nbon" in by_nb and "nboff" in by_nb
       for by_nb in records.values()):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for domain, by_nb in records.items():
        if "nbon" not in by_nb or "nboff" not in by_nb:
            continue
        on  = {tuple(r["grid"]): r for r in by_nb["nbon"]}
        off = {tuple(r["grid"]): r for r in by_nb["nboff"]}
        common = sorted(set(on) & set(off), key=lambda g: g[0] * g[1] * g[2])
        if not common:
            continue
        cells = np.array([g[0] * g[1] * g[2] for g in common], dtype=float)
        speedup = np.array([
            off[g]["cat_ms"]["TOTAL step"] / on[g]["cat_ms"]["TOTAL step"]
            for g in common
        ])
        ax.semilogx(cells, speedup, marker="o", linewidth=1.6,
                    label=DOMAIN_LABEL.get(domain, domain))

    ax.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=1.0,
               label="break-even (1×)")
    ax.set_xlabel("Total cells  $N_x N_y N_z$")
    ax.set_ylabel("Speed-up  (full-grid  /  narrow-band)")
    ax.set_title("Narrow-band speed-up vs body fraction")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "body_fraction_speedup.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Figure saved → {path}")
    plt.close(fig)


print(f"\n  Done. Figures saved to: {args.out_dir}")
