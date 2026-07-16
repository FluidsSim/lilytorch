#!/usr/bin/env python3
"""Compare simulated zebrafish kinematics and speed against real data.

Loads real kinematics (model_angles.xlsx) and keypoint trajectories from the
``keypoints/`` folder, and simulation output from ``simulation.hdf5``, then
plots joint-angle traces and forward/lateral swimming speed for visual
comparison.

Usage
-----
    python compare_sim_real.py /path/to/sim_output_dir [--speed fast|slow]

If ``--speed`` is omitted the script tries to infer it from the filename inside
the sim output directory.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from farms_core.sensors.sensor_convention import sc

from lilytorch.util.metrics import compute_speed_PCA

# ---------------------------------------------------------------------------
# Paths relative to this script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_KEYPOINTS_DIR = _SCRIPT_DIR  # this script lives inside keypoints/
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..")  # data_kinematics_control/

# Real data files per speed type
_REAL_FILES = {
    "fast": {
        "angles": "ep223_Cl1_fast_fish13_model_angles.xlsx",
        "xy_bl":  "ep223_Cl1_fast_fish13_XY_BL.csv",
    },
    "slow": {
        "angles": "ep248_Cl2_slow_fish13_model_angles.xlsx",
        "xy_bl":  "ep248_Cl2_slow_fish13_XY_BL.csv",
    },
}

# Model body length in metres
_MODEL_BL = 0.018  # from MODEL_POINTS_POSITIONS[-1] in extrapolate_angles.py

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "legend.fontsize":  8.5,
    "lines.linewidth":  1.4,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
})

# ---------------------------------------------------------------------------
# Helpers: real data
# ---------------------------------------------------------------------------

def load_real_kinematics(speed_type: str):
    """Return (time_s, angles_rad) for the real animal.

    Angles are the 15 model joint angles (Joint_0 … Joint_14) in radians.
    """
    fname = _REAL_FILES[speed_type]["angles"]
    # The model_angles xlsx is in keypoints/ and also in the parent dir.
    path_kp = os.path.join(_KEYPOINTS_DIR, fname)
    path_up = os.path.join(_DATA_DIR, fname)
    path = path_kp if os.path.exists(path_kp) else path_up

    df = pd.read_excel(path)
    times = df["time"].values.astype(float)
    angle_cols = [c for c in df.columns if c.startswith("Joint_")]
    angles = df[angle_cols].values.astype(float)
    return times, angles


def load_real_speed(speed_type: str):
    """Return (time_s, speed_bl_s) computed from head-point displacement.

    Speed is smoothed with a Gaussian kernel (σ ≈ 5 ms).
    """
    fname = _REAL_FILES[speed_type]["xy_bl"]
    path = os.path.join(_KEYPOINTS_DIR, fname)

    df = pd.read_csv(path)
    times = df["time_ms"].values.astype(float) / 1000.0

    # Head point = first tracked point
    x = df["x1_BL"].values.astype(float)
    y = df["y1_BL"].values.astype(float)

    # Finite-difference velocity (central differences, forward/backward at ends)
    dt = np.gradient(times)
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    speed_bl_s = np.sqrt(vx**2 + vy**2)

    # Light Gaussian smoothing
    from scipy.ndimage import gaussian_filter1d
    sigma = max(1, int(0.005 / np.median(dt)))  # ~5 ms
    speed_bl_s = gaussian_filter1d(speed_bl_s, sigma=sigma)

    return times, speed_bl_s


# ---------------------------------------------------------------------------
# Helpers: simulation data
# ---------------------------------------------------------------------------

def _find_sim_dir(candidate: str) -> str:
    """Resolve a simulation output directory containing simulation.hdf5."""
    if os.path.isdir(candidate):
        h5 = os.path.join(candidate, "output", "simulation.hdf5")
        if os.path.exists(h5):
            return candidate
    # Try glob if a partial path was given
    for d in sorted(glob.glob(candidate + "*"), reverse=True):
        h5 = os.path.join(d, "output", "simulation.hdf5")
        if os.path.exists(h5):
            return d
    raise FileNotFoundError(
        f"No simulation.hdf5 found under {candidate}"
    )


def load_sim_data(sim_dir: str):
    """Return (time_s, joint_angles_rad, v_forward_m_s, v_lateral_m_s).

    joint_angles : (nt, n_joints) – MuJoCo joint positions [rad]
    v_forward    : (nt,) – PCA forward speed [m/s]
    v_lateral    : (nt,) – PCA lateral speed [m/s]
    """
    h5_path = os.path.join(sim_dir, "output", "simulation.hdf5")
    with h5py.File(h5_path, "r") as f:
        # Joint positions: shape (nt, n_joints, 17), column 0 = position
        joints_arr = np.array(
            f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"]
        )
        joint_positions = joints_arr[:, :, sc.joint_position]  # (nt, n_joints)

        # Link positions & velocities for speed
        link_array = np.array(
            f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"]
        )
        links_vel = link_array[
            :, :, sc.link_com_velocity_lin_x : sc.link_com_velocity_lin_z + 1
        ]
        links_pos = link_array[
            :, :, sc.link_com_position_x : sc.link_com_position_z + 1
        ]

        if "times" in f:
            times = np.array(f["times"])[: link_array.shape[0]]
        else:
            timestep = float(np.array(f["timestep"]))
            times = timestep * np.arange(link_array.shape[0])

    # PCA speed
    v_fwd, v_lat = compute_speed_PCA(links_pos, links_vel)

    return times, joint_positions, np.asarray(v_fwd), np.asarray(v_lat)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _infer_speed_type(sim_dir: str) -> str:
    """Guess 'fast' or 'slow' from the sim directory or its config."""
    basename = os.path.basename(sim_dir).lower()
    # Check config yaml for the xlsx filename
    for yml_name in ["animat_config_0.yaml", "parameters.yaml"]:
        yml_path = os.path.join(sim_dir, yml_name)
        if os.path.exists(yml_path):
            import yaml
            with open(yml_path) as fh:
                cfg = yaml.unsafe_load(fh)
            # Try to find the angles file in the config
            cfg_str = str(cfg)
            if "ep223" in cfg_str or "fast" in cfg_str:
                return "fast"
            if "ep248" in cfg_str or "slow" in cfg_str:
                return "slow"
    if "fast" in basename:
        return "fast"
    if "slow" in basename:
        return "slow"
    raise ValueError(
        "Cannot infer speed type from sim dir.  Use --speed fast|slow."
    )


def plot_comparison(
    sim_dir: str,
    speed_type: str,
    out_path: str | None = None,
):
    """Main plotting routine."""

    # ── Load real data ───────────────────────────────────────────────
    t_real_ang, real_angles = load_real_kinematics(speed_type)
    t_real_spd, real_speed_bl_s = load_real_speed(speed_type)

    # ── Load simulation data ─────────────────────────────────────────
    t_sim, sim_joints, sim_v_fwd, sim_v_lat = load_sim_data(sim_dir)
    sim_speed_m_s = np.sqrt(sim_v_fwd**2 + sim_v_lat**2)
    sim_speed_bl_s = sim_speed_m_s / _MODEL_BL

    # ── Align time axes: start both at t=0 ──────────────────────────
    t_real_ang = t_real_ang - t_real_ang[0]
    t_real_spd = t_real_spd - t_real_spd[0]
    t_sim = t_sim - t_sim[0]

    # ── Match number of joints for comparison ────────────────────────
    n_real_joints = real_angles.shape[1]
    n_sim_joints  = sim_joints.shape[1]
    n_compare = min(n_real_joints, n_sim_joints)

    real_angles_trim = real_angles[:, :n_compare]
    sim_joints_trim  = sim_joints[:, :n_compare]

    # ── Create figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_ang_real, ax_ang_sim), (ax_speed, ax_traj) = axes

    title_suffix = f" ({speed_type.capitalize()})"

    # --- Joint angles: real ---
    for j in range(n_compare):
        ax_ang_real.plot(t_real_ang, real_angles_trim[:, j],
                         alpha=0.7, linewidth=0.8)
    ax_ang_real.set_title("Real Joint Angles" + title_suffix)
    ax_ang_real.set_xlabel("Time [s]")
    ax_ang_real.set_ylabel("Joint angle [rad]")
    ax_ang_real.set_xlim(0, t_real_ang[-1])

    # --- Joint angles: simulation ---
    for j in range(n_compare):
        ax_ang_sim.plot(t_sim, sim_joints_trim[:, j],
                        alpha=0.7, linewidth=0.8)
    ax_ang_sim.set_title("Simulated Joint Angles" + title_suffix)
    ax_ang_sim.set_xlabel("Time [s]")
    ax_ang_sim.set_ylabel("Joint angle [rad]")
    ax_ang_sim.set_xlim(0, t_sim[-1])

    # --- Speed comparison ---
    ax_speed.plot(t_real_spd, real_speed_bl_s,
                  label="Real (head displacement)", color="black", linewidth=1.8)
    ax_speed.plot(t_sim, sim_speed_bl_s,
                  label="Simulation (PCA COM speed)", color="#2196F3", linewidth=1.8)
    ax_speed.set_title("Swimming Speed" + title_suffix)
    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [BL/s]")
    ax_speed.legend(loc="upper right")
    ax_speed.set_xlim(0, max(t_real_spd[-1], t_sim[-1]))

    # --- Trajectory overlay ---
    # Real: head position in BL (shift to start at origin)
    fname = _REAL_FILES[speed_type]["xy_bl"]
    df = pd.read_csv(os.path.join(_KEYPOINTS_DIR, fname))
    x_head = df["x1_BL"].values.astype(float)
    y_head = df["y1_BL"].values.astype(float)
    x_head = x_head - x_head[0]
    y_head = y_head - y_head[0]

    # Sim: COM position from link data
    with h5py.File(os.path.join(sim_dir, "output", "simulation.hdf5"), "r") as f:
        link_array = np.array(
            f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"]
        )
        com_x = np.mean(
            link_array[:, :, sc.link_com_position_x], axis=1
        )
        com_y = np.mean(
            link_array[:, :, sc.link_com_position_y], axis=1
        )
    # Convert to BL and shift to origin
    com_x_bl = (com_x - com_x[0]) / _MODEL_BL
    com_y_bl = (com_y - com_y[0]) / _MODEL_BL

    ax_traj.plot(y_head, x_head, label="Real (head)", color="black", linewidth=1.5)
    ax_traj.plot(com_y_bl, com_x_bl, label="Simulation (COM)", color="#2196F3", linewidth=1.5)
    ax_traj.scatter(0, 0, marker="o", color="red", s=40, zorder=5, label="Start")
    ax_traj.set_title("Trajectory (top-down)" + title_suffix)
    ax_traj.set_xlabel("Lateral displacement [BL]")
    ax_traj.set_ylabel("Forward displacement [BL]")
    ax_traj.legend(loc="upper left")
    ax_traj.set_aspect("equal")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path)
        print(f"Saved comparison plot to {out_path}")
    else:
        out_default = os.path.join(sim_dir, f"comparison_{speed_type}.png")
        fig.savefig(out_default)
        print(f"Saved comparison plot to {out_default}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare simulated vs real zebrafish kinematics and speed"
    )
    parser.add_argument(
        "sim_dir",
        help="Path to simulation output directory (contains output/simulation.hdf5)",
    )
    parser.add_argument(
        "--speed",
        choices=["fast", "slow"],
        default=None,
        help="Speed type (fast=ep223, slow=ep248). Inferred if omitted.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <sim_dir>/comparison_<speed>.png)",
    )
    args = parser.parse_args()

    sim_dir = _find_sim_dir(args.sim_dir)
    speed_type = args.speed or _infer_speed_type(sim_dir)
    print(f"Sim dir:    {sim_dir}")
    print(f"Speed type: {speed_type}")

    plot_comparison(sim_dir, speed_type, args.out)


if __name__ == "__main__":
    main()
