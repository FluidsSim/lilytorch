#!/usr/bin/env python3
"""Analyze standalone (no-MuJoCo) Coquerelle FSI runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path("/data/andreaferrario/ns_data/coquerelle_adhoc_study")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CASES = [
    "heun_dt_0p00005",
    "euler_dt_0p00005",
    "euler_dt_0p000025",
    "euler_dt_0p0000125",
]


def load_case(case_name: str):
    cdir = OUT_ROOT / case_name
    summ = cdir / "adhoc_summary.json"
    ts = cdir / "adhoc_timeseries.npz"
    if not (summ.exists() and ts.exists()):
        print(f"[skip] missing outputs for {case_name}")
        return None
    with summ.open("r", encoding="utf-8") as f:
        s = json.load(f)
    arr = np.load(ts)
    return s, arr


def main():
    rows = []
    series = {}
    for c in CASES:
        out = load_case(c)
        if out is None:
            continue
        s, a = out
        rows.append(s)
        series[c] = a

    if not rows:
        print("No standalone runs found.")
        return

    out_csv = FIG_DIR / "coquerelle_adhoc_summary.csv"
    keys = [
        "case_name", "dt", "nt", "heun", "t_end", "y_end", "vy_end",
        "peak_vy", "t_peak", "y_peak",
        "vy_mean_y_1p5_1p0", "vy_mean_y_1p2_0p8", "vy_mean_y_1p0_0p6",
        "vy_tail_mean",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[saved] {out_csv}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in CASES:
        if c not in series:
            continue
        a = series[c]
        ax.plot(a["t"], a["vy"], label=c)
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("v_y [cm/s]")
    ax.set_title("Standalone Coquerelle FSI (no MuJoCo): v_y(t)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    out_png = FIG_DIR / "coquerelle_adhoc_vy_vs_t.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")

    print("\nSummary:")
    rows_sorted = sorted(rows, key=lambda r: (not r["heun"], r["dt"]))
    for r in rows_sorted:
        print(
            f"- {r['case_name']}: dt={r['dt']:.8f}, heun={r['heun']}, "
            f"peak_vy={r['peak_vy']:.6f}, mean_vy[y in 1.2..0.8]={r['vy_mean_y_1p2_0p8']:.6f}, "
            f"tail={r['vy_tail_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
