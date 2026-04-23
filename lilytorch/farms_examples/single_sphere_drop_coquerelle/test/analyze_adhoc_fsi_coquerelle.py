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

CASE_PREFIXES = (
    "test_",
    "heunFluid_",
    "eulerFluid_",
    "heun_dt_",
    "euler_dt_",
)

# Focused diagnostic groups from the latest integrator investigation.
PLOT_GROUPS = {
    "coquerelle_drag_only_vy_vs_t.png": {
        "title": "Drag-Only Isolation: Sphere v_y(t)",
        "cases": [
            "test_drag_heunFluid_bodyEuler_dt_0p00005",
            "test_drag_eulerFluid_bodyEuler_dt_0p00005",
        ],
    },
    "coquerelle_heun_force_staging_vy_vs_t.png": {
        "title": "Heun Fluid: Single vs Double Force Staging",
        "cases": [
            "test_full_heunFluid_bodyHeun_forceSingle_dt_0p00005",
            "test_full_heunFluid_bodyHeun_forceDouble_dt_0p00005",
        ],
    },
    "coquerelle_euler_force_staging_vy_vs_t.png": {
        "title": "Euler Fluid: Single vs Double Force Staging",
        "cases": [
            "test_full_eulerFluid_bodyHeun_forceSingle_dt_0p00005",
            "test_full_eulerFluid_bodyHeun_forceDouble_dt_0p00005",
        ],
    },
    "coquerelle_full_double_fluid_compare_vy_vs_t.png": {
        "title": "Full Forces, Double Staging: Heun vs Euler Fluid",
        "cases": [
            "test_full_heunFluid_bodyHeun_forceDouble_dt_0p00005",
            "test_full_eulerFluid_bodyHeun_forceDouble_dt_0p00005",
        ],
    },
}

CASE_LABELS = {
    "test_drag_heunFluid_bodyEuler_dt_0p00005": "drag-only, Heun fluid",
    "test_drag_eulerFluid_bodyEuler_dt_0p00005": "drag-only, Euler fluid",
    "test_full_heunFluid_bodyHeun_forceSingle_dt_0p00005": "Heun fluid, single force",
    "test_full_heunFluid_bodyHeun_forceDouble_dt_0p00005": "Heun fluid, double force",
    "test_full_eulerFluid_bodyHeun_forceSingle_dt_0p00005": "Euler fluid, single force",
    "test_full_eulerFluid_bodyHeun_forceDouble_dt_0p00005": "Euler fluid, double force",
}


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


def discover_cases():
    if not OUT_ROOT.exists():
        return []

    case_names = []
    for child in sorted(OUT_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(CASE_PREFIXES):
            continue
        if (child / "adhoc_summary.json").exists() and (child / "adhoc_timeseries.npz").exists():
            case_names.append(child.name)
    return case_names


def _label(case_name: str) -> str:
    return CASE_LABELS.get(case_name, case_name)


def _plot_case_group(series: dict, case_names: list[str], title: str, out_name: str):
    present = [c for c in case_names if c in series]
    if not present:
        print(f"[skip] no cases found for {out_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in present:
        a = series[c]
        ax.plot(a["t"], a["vy"], label=_label(c), linewidth=1.8)

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Sphere v_y [cm/s]")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _plot_compact_report(series: dict):
    panels = [
        (
            "Drag-Only Isolation",
            [
                "test_drag_heunFluid_bodyEuler_dt_0p00005",
                "test_drag_eulerFluid_bodyEuler_dt_0p00005",
            ],
        ),
        (
            "Heun Fluid: Single vs Double",
            [
                "test_full_heunFluid_bodyHeun_forceSingle_dt_0p00005",
                "test_full_heunFluid_bodyHeun_forceDouble_dt_0p00005",
            ],
        ),
        (
            "Euler Fluid: Single vs Double",
            [
                "test_full_eulerFluid_bodyHeun_forceSingle_dt_0p00005",
                "test_full_eulerFluid_bodyHeun_forceDouble_dt_0p00005",
            ],
        ),
        (
            "Double Staging: Heun vs Euler",
            [
                "test_full_heunFluid_bodyHeun_forceDouble_dt_0p00005",
                "test_full_eulerFluid_bodyHeun_forceDouble_dt_0p00005",
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes_flat = axes.ravel()

    for ax, (title, case_names) in zip(axes_flat, panels):
        present = [c for c in case_names if c in series]
        if not present:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("Sphere v_y [cm/s]")
            continue

        for c in present:
            a = series[c]
            ax.plot(a["t"], a["vy"], label=_label(c), linewidth=1.7)

        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Sphere v_y [cm/s]")
        ax.legend(framealpha=0.9, fontsize=8)

    fig.suptitle("Coquerelle Integrator Diagnostics: Sphere Velocity vs Time", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = FIG_DIR / "coquerelle_velocity_report_vy_vs_t.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    cases = discover_cases()
    if not cases:
        print("No standalone runs found.")
        return

    rows = []
    series = {}
    for c in cases:
        out = load_case(c)
        if out is None:
            continue
        s, a = out
        rows.append(s)
        series[c] = a

    out_csv = FIG_DIR / "coquerelle_adhoc_summary.csv"
    keys = [
        "case_name", "dt", "nt", "heun", "t_end", "y_end", "vy_end",
        "body_integrator", "force_model", "heun_force_update", "drag_coeff",
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
    for c in cases:
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

    for out_name, cfg in PLOT_GROUPS.items():
        _plot_case_group(series, cfg["cases"], cfg["title"], out_name)

    _plot_compact_report(series)

    print("\nSummary:")
    rows_sorted = sorted(rows, key=lambda r: (not r["heun"], r.get("body_integrator", "euler"), r["dt"]))
    for r in rows_sorted:
        print(
            f"- {r['case_name']}: dt={r['dt']:.8f}, heun={r['heun']}, "
            f"body_integrator={r.get('body_integrator', 'euler')}, "
            f"force_model={r.get('force_model', 'full')}, "
            f"heun_force_update={r.get('heun_force_update', 'double')}, "
            f"peak_vy={r['peak_vy']:.6f}, mean_vy[y in 1.2..0.8]={r['vy_mean_y_1p2_0p8']:.6f}, "
            f"tail={r['vy_tail_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
