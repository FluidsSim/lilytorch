#!/usr/bin/env python3
"""Analyze matched Heun/Euler Coquerelle runs.

Reads case outputs from /data/andreaferrario/ns_data/coquerelle_heun_euler_study
and generates:
  - figures/coquerelle_heun_euler_uz_vs_t.png
  - figures/coquerelle_heun_euler_z_vs_t.png
  - figures/coquerelle_heun_euler_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

OUT_ROOT = Path("/data/andreaferrario/ns_data/coquerelle_heun_euler_study")

CASES = [
    {"name": "heun_dt_0p00005", "label": "Heun dt=5e-5"},
    {"name": "heun_dt_0p000025", "label": "Heun dt=2.5e-5"},
    {"name": "euler_dt_0p00005", "label": "Euler dt=5e-5"},
    {"name": "euler_dt_0p000025", "label": "Euler dt=2.5e-5"},
    {"name": "euler_dt_0p0000125", "label": "Euler dt=1.25e-5"},
]


def load_case(case_name: str) -> dict | None:
    case_dir = OUT_ROOT / case_name
    sim_h5 = case_dir / "output" / "simulation.hdf5"
    if not sim_h5.exists():
        print(f"[skip] Missing file: {sim_h5}")
        return None

    data = hdf5_to_dict(str(sim_h5))
    times = np.asarray(data["times"])[:-1]
    sa = np.asarray(data["animats"][0]["sensors"]["links"]["array"][:, 0, :])

    pos = sa[:, sc.link_com_position_x : sc.link_com_position_z + 1]
    vel = sa[:, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]

    z = pos[:, 2]
    uz = vel[:, 2]

    dt_data = float(np.median(np.diff(times))) if len(times) > 1 else float("nan")

    i_peak = int(np.argmin(uz))
    peak = float(uz[i_peak])

    tail_start = int(0.9 * len(uz))
    tail_mean = float(np.mean(uz[tail_start:])) if len(uz) > 0 else float("nan")

    def window_mean(lo: float, hi: float) -> float:
        mask = (z <= lo) & (z >= hi)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(uz[mask]))

    return {
        "case_name": case_name,
        "times": times,
        "z": z,
        "uz": uz,
        "dt_data": dt_data,
        "t_end": float(times[-1]) if len(times) else float("nan"),
        "z_end": float(z[-1]) if len(z) else float("nan"),
        "uz_end": float(uz[-1]) if len(uz) else float("nan"),
        "peak_uz": peak,
        "t_peak": float(times[i_peak]),
        "z_peak": float(z[i_peak]),
        "uz_tail_mean": tail_mean,
        "uz_mean_z_1p5_1p0": window_mean(1.5, 1.0),
        "uz_mean_z_1p2_0p8": window_mean(1.2, 0.8),
        "uz_mean_z_1p0_0p6": window_mean(1.0, 0.6),
    }


def save_summary(rows: list[dict]) -> None:
    out_csv = FIG_DIR / "coquerelle_heun_euler_summary.csv"
    fields = [
        "case_name",
        "dt_data",
        "t_end",
        "z_end",
        "peak_uz",
        "t_peak",
        "z_peak",
        "uz_mean_z_1p5_1p0",
        "uz_mean_z_1p2_0p8",
        "uz_mean_z_1p0_0p6",
        "uz_tail_mean",
        "uz_end",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"[saved] {out_csv}")


def plot_uz(rows: list[dict], labels: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in rows:
        ax.plot(row["times"], row["uz"], label=labels.get(row["case_name"], row["case_name"]))
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("u_z [cm/s]")
    ax.set_title("Coquerelle: sphere vertical velocity")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    out_png = FIG_DIR / "coquerelle_heun_euler_uz_vs_t.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_z(rows: list[dict], labels: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in rows:
        ax.plot(row["times"], row["z"], label=labels.get(row["case_name"], row["case_name"]))
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [cm]")
    ax.set_title("Coquerelle: sphere z-position")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    out_png = FIG_DIR / "coquerelle_heun_euler_z_vs_t.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    labels = {c["name"]: c["label"] for c in CASES}

    rows = []
    for case in CASES:
        result = load_case(case["name"])
        if result is not None:
            rows.append(result)

    if not rows:
        print("No completed runs found. Run study cases first.")
        return

    save_summary(rows)
    plot_uz(rows, labels)
    plot_z(rows, labels)

    print("\nSummary (key metrics):")
    for row in rows:
        print(
            f"- {row['case_name']}: dt={row['dt_data']:.8f}, "
            f"peak_uz={row['peak_uz']:.6f}, "
            f"mean_uz[z in 1.2..0.8]={row['uz_mean_z_1p2_0p8']:.6f}, "
            f"tail_mean={row['uz_tail_mean']:.6f}, "
            f"final_uz={row['uz_end']:.6f}"
        )


if __name__ == "__main__":
    main()
