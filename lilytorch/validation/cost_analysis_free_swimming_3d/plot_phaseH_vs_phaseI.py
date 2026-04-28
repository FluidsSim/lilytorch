#!/usr/bin/env python3
"""Overlay loglog scaling: Phase H (fused set_BCs) vs Phase I (fused bdim_apply).

Reads cost_breakdown_*.csv from
  figures/scaling_conditions_phaseH/nbforces/
  figures/scaling_conditions_phaseI/nbforces/
and writes
  figures/scaling_conditions_phaseI/cost_scaling_phaseH_vs_phaseI.{pdf,png}
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASES = {
    "phaseH": dict(label="Phase H  (fused set_BCs)", color="#37474f", marker="s",
                   ls="--"),
    "phaseI": dict(label="Phase I  (fused bdim_apply)", color="#b71c1c", marker="o",
                   ls="-"),
}

_GRID_RE = re.compile(r"cost_breakdown_(\d+)x(\d+)x(\d+)\.csv")


def _load(phase):
    pts = []
    pat = os.path.join(
        SCRIPT_DIR, "figures", f"scaling_conditions_{phase}", "nbforces",
        "cost_breakdown_*.csv",
    )
    for f in sorted(glob.glob(pat)):
        m = _GRID_RE.search(os.path.basename(f))
        if not m:
            continue
        nx, ny, nz = (int(x) for x in m.groups())
        cells = nx * ny * nz
        for row in csv.DictReader(open(f)):
            if row["component"] == "TOTAL step":
                pts.append((cells, float(row["mean_ms"])))
                break
    pts.sort()
    return pts


fig, ax = plt.subplots(figsize=(7.0, 4.6))

all_x = []
for phase, style in PHASES.items():
    pts = _load(phase)
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    all_x.extend(xs)
    ax.loglog(xs, ys, marker=style["marker"], color=style["color"],
              linestyle=style["ls"], linewidth=1.6, markersize=7,
              label=style["label"])

# Linear-scaling reference passing through smallest Phase I point
pts = _load("phaseI")
if pts:
    x0, y0 = pts[0]
    xref = sorted(set(all_x))
    yref = [y0 * (x / x0) for x in xref]
    ax.loglog(xref, yref, color="#9e9e9e", linestyle=":", linewidth=1.0,
              label=r"linear scaling $\propto N_xN_yN_z$")

ax.set_xlabel(r"Total cells  $N_x N_y N_z$")
ax.set_ylabel("Time per step (ms)")
ax.set_title("Phase H vs Phase I — nbforces (streaming + fused forces)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", framealpha=0.92, fontsize=9)
fig.tight_layout()

out_pdf = os.path.join(
    SCRIPT_DIR, "figures", "scaling_conditions_phaseI",
    "cost_scaling_phaseH_vs_phaseI.pdf",
)
fig.savefig(out_pdf)
fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150)
plt.close(fig)
print(f"Saved → {out_pdf}")
print(f"Saved → {out_pdf.replace('.pdf', '.png')}")

# Print speedup table
print()
print(f"{'grid (cells)':>14}  {'phaseH ms':>10}  {'phaseI ms':>10}  {'speedup':>8}")
H = dict(_load("phaseH"))
I = dict(_load("phaseI"))
for c in sorted(set(H) & set(I)):
    sp = H[c] / I[c]
    print(f"{c:>14,}  {H[c]:>10.3f}  {I[c]:>10.3f}  {sp:>7.3f}x")
