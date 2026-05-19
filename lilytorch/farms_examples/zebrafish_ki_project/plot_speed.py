"""Plot forward and lateral COM speed of the zebrafish KI-project runs.

Usage
-----
# single run
python plot_speed.py /data/andreaferrario/ns_data/2026-05-19T16:04:23.075402

# multiple runs overlaid
python plot_speed.py /data/andreaferrario/ns_data/run_slow /data/andreaferrario/ns_data/run_fast

# all time-stamped sub-dirs inside a stack folder
python plot_speed.py --stack /data/andreaferrario/ns_data/zebrafish_ki

Options
-------
--bl        Body length in metres used for normalisation (default: 0.004 m = 4 mm)
--out       Output PNG path (default: speed_plot.png next to the first run dir)
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from farms_core.sensors.sensor_convention import sc

from lilytorch.util.metrics import compute_speed_PCA

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  13,
    "axes.titlesize":  13,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.6,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linewidth":  0.5,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def _label_for_dir(run_dir: str) -> str:
    """Try to build a human-readable label from animat_config_0.yaml."""
    cfg_path = os.path.join(run_dir, "animat_config_0.yaml")
    if not os.path.exists(cfg_path):
        return os.path.basename(run_dir)
    try:
        with open(cfg_path) as f:
            cfg = yaml.unsafe_load(f)
        # muscle controller config is in extensions[0].config
        ext_cfg = cfg.get("extensions", [{}])[0].get("config", {})
        mode = ext_cfg.get("mode")
        if mode is not None:
            return f"mode={mode}"
    except Exception:
        pass
    return os.path.basename(run_dir)


def _load_com_velocity(run_dir: str):
    """Return (times, v_forward, v_lateral) PCA-projected speed arrays.

    The forward axis is the principal axis of the body (head→tail direction)
    at each time step; lateral is the perpendicular left-pointing axis.
    """
    hdf5_path = os.path.join(run_dir, "output", "simulation.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"simulation.hdf5 not found in {run_dir}/output/")

    with h5py.File(hdf5_path, "r") as f:
        link_array = np.array(
            f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"]
        )
        if "times" in f:
            times = np.array(f["times"])[: link_array.shape[0]]
        else:
            timestep = float(np.array(f["timestep"]))
            times = timestep * np.arange(link_array.shape[0])

    links_vel = link_array[
        :, :, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1
    ]  # (nt, n_links, 3)
    links_pos = link_array[
        :, :, sc.link_com_position_x : sc.link_com_position_z + 1
    ]  # (nt, n_links, 3)

    v_forward, v_lateral = compute_speed_PCA(links_pos, links_vel)

    return times, np.asarray(v_forward), np.asarray(v_lateral)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot zebrafish forward/lateral COM speed.")
    parser.add_argument(
        "dirs",
        nargs="*",
        help="One or more run directories (each containing output/simulation.hdf5).",
    )
    parser.add_argument(
        "--stack",
        default=None,
        metavar="STACK_DIR",
        help="Stack folder: all immediate sub-directories are treated as runs.",
    )
    parser.add_argument(
        "--bl",
        type=float,
        default=0.02,
        help="Body length in metres for normalisation (default: 0.02 m).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: speed_plot.png next to the first run).",
    )
    args = parser.parse_args()

    run_dirs: list[str] = list(args.dirs)

    if args.stack:
        sub = sorted(
            os.path.join(args.stack, d)
            for d in os.listdir(args.stack)
            if os.path.isdir(os.path.join(args.stack, d))
        )
        run_dirs = sub + run_dirs

    if not run_dirs:
        parser.error("Provide at least one run directory or use --stack.")

    BL = args.bl
    out_path = args.out or os.path.join(run_dirs[0], "speed_plot.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for run_dir in run_dirs:
        try:
            times, v_x, v_y = _load_com_velocity(run_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        label = _label_for_dir(run_dir)
        ax1.plot(times, v_x / BL, label=label)
        ax2.plot(times, v_y / BL, label=label)

    ax1.set_ylabel("$V_{\\mathrm{fwd}}$ [BL/s]")
    ax1.set_title("Forward speed (PCA body axis)")
    ax1.legend()

    ax2.set_ylabel("$V_{\\mathrm{lat}}$ [BL/s]")
    ax2.set_title("Lateral speed (perpendicular to body axis)")
    ax2.set_xlabel("Time [s]")
    ax2.legend()

    fig.suptitle(f"Zebrafish KI — COM speed  (BL = {BL * 1e3:.1f} mm)", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
