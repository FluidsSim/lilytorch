"""Plot 3-D dropped-sphere study results and compare with Coquerelle-Cottet.

Outputs:
  1. Sphere height histories for available cases.
  2. Downward settling-speed histories for available cases.
  3. Terminal-speed comparison against Table 1 of Coquerelle & Cottet (2008).
  4. CSV summary with paper values and measured run statistics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_ROOT = Path("/data/andreaferrario/ns_data/coquerelle_cottet_3d_drop")
FMT = ".svg"

PAPER_BOX = (0.0, 1.0, 0.0, 1.0, 0.0, 4.0)

VISCOSITY_COLORS = {
    0.10: "#1b9e77",
    0.05: "#d95f02",
    0.02: "#7570b3",
}


@dataclass(frozen=True)
class PaperCase:
    name: str
    diameter: float
    viscosity: float
    u_exp: float
    u_h0: float
    u_h_1_64: float
    u_glowinski: float


PAPER_CASES = [
    PaperCase("d0p2_nu0p1", 0.2, 0.10, 0.2571, 0.2750, 0.2560, 0.2567),
    PaperCase("d0p2_nu0p05", 0.2, 0.05, 0.4603, 0.5130, 0.4750, 0.4844),
    PaperCase("d0p2_nu0p02", 0.2, 0.02, 0.9129, 1.0160, 0.9370, 0.9480),
    PaperCase("d0p3_nu0p1", 0.3, 0.10, 0.4047, 0.4350, 0.4010, 0.4072),
    PaperCase("d0p3_nu0p05", 0.3, 0.05, 0.7493, 0.7950, 0.7480, 0.7599),
    PaperCase("d0p3_nu0p02", 0.3, 0.02, 1.4359, 1.5100, 1.3900, 1.3920),
]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def load_hdf5_kinematics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from farms_core.io.hdf5 import hdf5_to_dict
    from farms_core.sensors.sensor_convention import sc

    data = hdf5_to_dict(str(path))
    times = np.asarray(data["times"][:-1], dtype=float)
    sensor_array = np.asarray(data["animats"][0]["sensors"]["links"]["array"][:, 0, :], dtype=float)
    com_pos = sensor_array[:, sc.link_com_position_x : sc.link_com_position_z + 1]
    com_vel = sensor_array[:, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1]
    return times, com_pos, com_vel


def build_case_summary(case: PaperCase) -> dict:
    sim_cfg_path = HERE / f"simulation_config_{case.name}.yaml"
    animat_cfg_path = HERE / ("sphere_D0p2.yaml" if case.diameter == 0.2 else "sphere_D0p3.yaml")
    hdf5_path = OUT_ROOT / case.name / "output" / "simulation.hdf5"

    sim_cfg = read_yaml(sim_cfg_path) if sim_cfg_path.exists() else None
    animat_cfg = read_yaml(animat_cfg_path) if animat_cfg_path.exists() else None

    solver = None
    if sim_cfg is not None:
        for ext in sim_cfg.get("extensions", []):
            if ext.get("loader") == "lilytorch.integration.extensions.FluidExtension":
                solver = ext["config"]["bdim_yaml"]["solver"]
                break

    z0 = None
    if animat_cfg is not None:
        z0 = float(animat_cfg["spawn"]["pose"][2])

    summary = {
        "name": case.name,
        "diameter": case.diameter,
        "viscosity": case.viscosity,
        "u_exp": case.u_exp,
        "u_h0": case.u_h0,
        "u_h_1_64": case.u_h_1_64,
        "u_glowinski": case.u_glowinski,
        "available": hdf5_path.exists(),
        "path": hdf5_path,
        "times": None,
        "z": None,
        "u_settle": None,
        "dt": float(sim_cfg["physics"]["timestep"]) if sim_cfg is not None else np.nan,
        "z0_config": z0 if z0 is not None else np.nan,
        "zmax_config": float(solver["zmax"]) if solver is not None else np.nan,
        "terminal_mean": np.nan,
        "terminal_std": np.nan,
        "final_speed": np.nan,
        "peak_speed": np.nan,
        "final_z": np.nan,
        "sample_count": 0,
        "rel_error_exp_pct": np.nan,
        "rel_error_h0_pct": np.nan,
    }

    if not hdf5_path.exists():
        return summary

    times, com_pos, com_vel = load_hdf5_kinematics(hdf5_path)
    z = com_pos[:, 2]
    u_settle = -com_vel[:, 2]

    radius = 0.5 * case.diameter
    clearance_mask = z > max(2.5 * radius, 0.0)
    if not np.any(clearance_mask):
        clearance_mask = np.ones_like(z, dtype=bool)

    t_end = float(times[-1]) if len(times) else 0.0
    window_seconds = min(0.5, 0.2 * t_end) if t_end > 0 else 0.0
    tail_mask = times >= (t_end - window_seconds)
    estimate_mask = clearance_mask & tail_mask
    if np.count_nonzero(estimate_mask) < 25:
        estimate_mask = clearance_mask & (times >= (t_end - 0.1 * t_end))
    if np.count_nonzero(estimate_mask) < 5:
        estimate_mask = clearance_mask

    terminal_mean = float(np.mean(u_settle[estimate_mask]))
    terminal_std = float(np.std(u_settle[estimate_mask]))

    summary.update({
        "times": times,
        "z": z,
        "u_settle": u_settle,
        "terminal_mean": terminal_mean,
        "terminal_std": terminal_std,
        "final_speed": float(u_settle[-1]),
        "peak_speed": float(np.max(u_settle)),
        "final_z": float(z[-1]),
        "sample_count": int(np.count_nonzero(estimate_mask)),
        "rel_error_exp_pct": 100.0 * (terminal_mean - case.u_exp) / case.u_exp,
        "rel_error_h0_pct": 100.0 * (terminal_mean - case.u_h0) / case.u_h0,
    })
    return summary


def load_summaries() -> list[dict]:
    return [build_case_summary(case) for case in PAPER_CASES]


def print_setup_note(summaries: list[dict]) -> None:
    print("Paper reference: Coquerelle & Cottet (2008), Table 1 / Section 4.2.1")
    print("Paper 3-D box: [0,1] x [0,1] x [0,4]")
    print("Current study uses the generated configs in this folder.")
    if summaries:
        zmax = summaries[0]["zmax_config"]
        z0 = summaries[0]["z0_config"]
        print(f"Current config: zmax={zmax}, initial z={z0}")
        if not np.isnan(zmax) and abs(zmax - PAPER_BOX[-1]) > 1e-12:
            print("WARNING: current study box height differs from the paper (4.0).")
        print("The paper does not explicitly state the initial sphere position for Section 4.2.1.")


def write_summary_csv(summaries: list[dict]) -> Path:
    out_csv = FIG_DIR / "coquerelle_cottet_3d_summary.csv"
    fieldnames = [
        "name",
        "diameter",
        "viscosity",
        "available",
        "dt",
        "z0_config",
        "zmax_config",
        "terminal_mean",
        "terminal_std",
        "final_speed",
        "peak_speed",
        "final_z",
        "sample_count",
        "u_exp",
        "u_h0",
        "u_h_1_64",
        "u_glowinski",
        "rel_error_exp_pct",
        "rel_error_h0_pct",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key) for key in fieldnames}
            writer.writerow(row)
    return out_csv


def plot_height_histories(summaries: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    diameter_values = [0.2, 0.3]

    for ax, diameter in zip(axes, diameter_values):
        plotted_any = False
        for summary in summaries:
            if summary["diameter"] != diameter or not summary["available"]:
                continue
            ax.plot(
                summary["times"],
                summary["z"],
                color=VISCOSITY_COLORS[summary["viscosity"]],
                label=fr"$\nu={summary['viscosity']:.2f}$",
            )
            plotted_any = True
        ax.set_title(fr"$D={diameter:.1f}$")
        ax.set_xlabel("$t$")
        ax.axhline(0.5 * diameter, color="0.5", linestyle="--", linewidth=0.9)
        if not plotted_any:
            ax.text(0.5, 0.5, "No completed runs", transform=ax.transAxes,
                    ha="center", va="center", color="0.4")
        if plotted_any:
            ax.legend(framealpha=0.92, edgecolor="0.7")

    axes[0].set_ylabel("Sphere center height $z$")
    fig.suptitle("3-D sphere sedimentation: height histories")
    fig.tight_layout()
    out_path = FIG_DIR / f"coquerelle_cottet_3d_z_vs_t{FMT}"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_speed_histories(summaries: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    diameter_values = [0.2, 0.3]

    for ax, diameter in zip(axes, diameter_values):
        plotted_any = False
        for summary in summaries:
            if summary["diameter"] != diameter or not summary["available"]:
                continue
            ax.plot(
                summary["times"],
                summary["u_settle"],
                color=VISCOSITY_COLORS[summary["viscosity"]],
                label=fr"$\nu={summary['viscosity']:.2f}$",
            )
            plotted_any = True
        ax.set_title(fr"$D={diameter:.1f}$")
        ax.set_xlabel("$t$")
        if not plotted_any:
            ax.text(0.5, 0.5, "No completed runs", transform=ax.transAxes,
                    ha="center", va="center", color="0.4")
        if plotted_any:
            ax.legend(framealpha=0.92, edgecolor="0.7")

    axes[0].set_ylabel("Downward speed $U=-u_z$")
    fig.suptitle("3-D sphere sedimentation: settling-speed histories")
    fig.tight_layout()
    out_path = FIG_DIR / f"coquerelle_cottet_3d_u_vs_t{FMT}"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_terminal_comparison(summaries: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(summaries), dtype=float)

    labels = [fr"$\nu={s['viscosity']:.2f}$" for s in summaries]

    ax.plot(x, [s["u_exp"] for s in summaries], marker="o", linestyle="none", color="#1b9e77", label="exp")
    ax.plot(x, [s["u_h_1_64"] for s in summaries], marker="v", linestyle="none", color="#e7298a", label="coquerelle & cotet")

    available_x = [idx for idx, s in enumerate(summaries) if s["available"]]
    available_y = [s["terminal_mean"] for s in summaries if s["available"]]
    if available_x:
        ax.plot(
            available_x,
            available_y,
            marker="D",
            linestyle="none",
            markersize=7,
            color="black",
            label="ours",
            zorder=5,
        )

    ax.axvline(2.5, color="0.75", linewidth=1.0)
    ax.text(1.25, 0.97, "$D=0.2$", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color="0.35", fontsize=11)
    ax.text(4.25, 0.97, "$D=0.3$", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color="0.35", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Terminal settling speed")
    ax.set_title("Coquerelle-Cottet Table 1 comparison")
    ax.legend(ncol=2, framealpha=0.92, edgecolor="0.7")

    fig.text(
        0.01,
        -0.02,
        "Note: the paper states the 3-D box as [0,1] x [0,1] x [0,4]. "
        "This example folder currently uses the generated study configs, which differ in box height.",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = FIG_DIR / f"coquerelle_cottet_3d_terminal_comparison{FMT}"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def print_summary_table(summaries: list[dict]) -> None:
    print("\nCase summary:")
    for summary in summaries:
        if summary["available"]:
            print(
                f"  {summary['name']}: U_tail={summary['terminal_mean']:.4f} "
                f"(+/- {summary['terminal_std']:.4f}), U_exp={summary['u_exp']:.4f}, "
                f"U_h0={summary['u_h0']:.4f}, err_exp={summary['rel_error_exp_pct']:+.1f}%"
            )
        else:
            print(
                f"  {summary['name']}: no simulation.hdf5 found; "
                f"paper U_exp={summary['u_exp']:.4f}, U_h0={summary['u_h0']:.4f}"
            )


def main() -> None:
    summaries = load_summaries()
    print_setup_note(summaries)
    out_csv = write_summary_csv(summaries)
    z_plot = plot_height_histories(summaries)
    u_plot = plot_speed_histories(summaries)
    cmp_plot = plot_terminal_comparison(summaries)
    print_summary_table(summaries)
    print("\nSaved:")
    print(f"  {out_csv}")
    print(f"  {z_plot}")
    print(f"  {u_plot}")
    print(f"  {cmp_plot}")


if __name__ == "__main__":
    main()