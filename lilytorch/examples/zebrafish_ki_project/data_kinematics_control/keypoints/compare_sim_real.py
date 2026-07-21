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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline, interp1d

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




def _instantaneous_speed(times, x, y):
    """Instantaneous speed: |COM(t) − COM(t−1)| / dt.

    Simple adjacent-frame finite difference.

    Parameters
    ----------
    times : (nt,) array — seconds
    x, y  : (nt,) arrays — COM positions (any consistent units)

    Returns
    -------
    speed : (nt,) array — same units as x, y per second
    """
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dt = np.diff(times, prepend=times[1] - times[0] if len(times) > 1 else 1.0)
    speed = np.sqrt(dx**2 + dy**2)
    mask = dt > 0
    speed[mask] = speed[mask] / dt[mask]
    return speed


def load_real_speed(speed_type: str):
    """Return (time_s, speed_bl_s) — instantaneous COM speed.

    Uses the centroid of all tracked keypoints as COM.  Speed is
    |COM(t) − COM(t−1)| / dt — a simple adjacent-frame estimate.
    """
    fname = _REAL_FILES[speed_type]["xy_bl"]
    path = os.path.join(_KEYPOINTS_DIR, fname)

    df = pd.read_csv(path)
    times = df["time_ms"].values.astype(float) / 1000.0

    x_cols = [c for c in df.columns if c.startswith("x")]
    y_cols = [c for c in df.columns if c.startswith("y")]
    x_com = df[x_cols].values.astype(float).mean(axis=1)
    y_com = df[y_cols].values.astype(float).mean(axis=1)

    speed_bl_s = _instantaneous_speed(times, x_com, y_com)
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
    t_sim, sim_joints, _sim_v_fwd, _sim_v_lat = load_sim_data(sim_dir)

    # Compute sim cumulative speed from COM — same gradient-free metric
    # as the real data, so the comparison is completely fair.
    with h5py.File(os.path.join(sim_dir, "output", "simulation.hdf5"), "r") as f:
        link_array = np.array(
            f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"]
        )
        com_x_sim = np.mean(link_array[:, :, sc.link_com_position_x], axis=1)
        com_y_sim = np.mean(link_array[:, :, sc.link_com_position_y], axis=1)
        if "times" in f:
            t_sim_pos = np.array(f["times"])[: link_array.shape[0]]
        else:
            t_sim_pos = t_sim
    sim_speed_bl_s = _instantaneous_speed(t_sim_pos, com_x_sim, com_y_sim) / _MODEL_BL

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
                  label="Real (instantaneous |Δ|/dt)", color="black", linewidth=1.8)
    ax_speed.plot(t_sim, sim_speed_bl_s,
                  label="Simulation (instantaneous |Δ|/dt)", color="#2196F3", linewidth=1.8)
    ax_speed.set_title("Instantaneous Speed" + title_suffix)
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

    # ── Second figure: aligned keypoint overlay ──────────────────────
    _plot_aligned_keypoints(
        sim_dir, speed_type, link_array, t_sim_pos,
        out_path=out_path, speed_label=speed_type,
    )

    # ── Third figure: COM positions 2D (all iterations) ──────────────
    _plot_com_positions_2d(
        sim_dir, speed_type, link_array, t_sim_pos,
        out_path=out_path, speed_label=speed_type,
    )



# ---------------------------------------------------------------------------
# Aligned keypoint overlay (second figure)
# ---------------------------------------------------------------------------

# Model points definition (from extrapolate_angles.py)
_MODEL_PTS = np.array([
    0.000, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018,
])
_MODEL_PTS_REL = _MODEL_PTS / _MODEL_PTS[-1]
_N_MODEL = len(_MODEL_PTS_REL)
_N_PASSIVE = 2
_N_ACTIVE = _N_MODEL - _N_PASSIVE


def _spline_model_points(x_kp, y_kp):
    """Map raw keypoints → 17 model points via cubic spline (same as extrapolate_angles.py)."""
    pos_vals = np.stack([x_kp, y_kp], axis=1)
    links_len = np.linalg.norm(np.diff(pos_vals, axis=0), axis=1)
    links_frac = np.cumsum(links_len) / np.sum(links_len)
    pts_frac = np.concatenate([[0], links_frac])
    arc0 = _MODEL_PTS_REL[1]
    arc1 = _MODEL_PTS_REL[_N_ACTIVE - 1]
    pts_arc = arc0 + pts_frac * (arc1 - arc0)

    spline_x = CubicSpline(pts_arc, x_kp, extrapolate=False)
    spline_y = CubicSpline(pts_arc, y_kp, extrapolate=False)
    smin, smax = pts_arc[0], pts_arc[-1]

    mx = np.zeros(_N_MODEL)
    my = np.zeros(_N_MODEL)
    for i, s in enumerate(_MODEL_PTS_REL):
        if s < smin:
            mx[i] = spline_x(smin) + spline_x(smin, 1) * (s - smin)
            my[i] = spline_y(smin) + spline_y(smin, 1) * (s - smin)
        elif s > smax:
            mx[i] = spline_x(smax) + spline_x(smax, 1) * (s - smax)
            my[i] = spline_y(smax) + spline_y(smax, 1) * (s - smax)
        else:
            mx[i] = spline_x(s)
            my[i] = spline_y(s)
    return mx, my


def _plot_aligned_keypoints(sim_dir, speed_type, link_array, t_sim,
                             out_path=None, speed_label="slow", n_times=12):
    """Generate a second figure: aligned keypoint overlay with time colorbar."""

    pos_x = link_array[:, :, sc.link_com_position_x]
    pos_y = link_array[:, :, sc.link_com_position_y]
    quat_x = link_array[:, 0, sc.link_com_orientation_x]
    quat_y = link_array[:, 0, sc.link_com_orientation_y]
    quat_z = link_array[:, 0, sc.link_com_orientation_z]
    quat_w = link_array[:, 0, sc.link_com_orientation_w]

    sim_init_yaw = np.arctan2(
        2 * (quat_w[0] * quat_z[0] + quat_x[0] * quat_y[0]),
        1 - 2 * (quat_y[0]**2 + quat_z[0]**2),
    )

    # ── Real keypoints ───────────────────────────────────────────────
    fname = _REAL_FILES[speed_type]["xy_bl"]
    csv_path = os.path.join(_KEYPOINTS_DIR, fname)
    df = pd.read_csv(csv_path)
    real_times = df["time_ms"].values.astype(float) / 1000.0
    x_cols = [c for c in df.columns if c.startswith("x")]
    y_cols = [c for c in df.columns if c.startswith("y")]
    x_vals = df[x_cols].values
    y_vals = df[y_cols].values

    exp_init_yaw = np.arctan2(
        y_vals[0, 1] - y_vals[0, 0],
        x_vals[0, 1] - x_vals[0, 0],
    )
    rot_angle = sim_init_yaw - exp_init_yaw

    # ── Common time points ───────────────────────────────────────────
    sim_times = t_sim
    t_min = max(sim_times[0], real_times[0])
    t_max = min(sim_times[-1], real_times[-1])
    common_times = np.linspace(t_min, t_max, n_times)

    sim_x_at_t = np.zeros((n_times, 16))
    sim_y_at_t = np.zeros((n_times, 16))
    for link_i in range(16):
        sim_x_at_t[:, link_i] = interp1d(sim_times, pos_x[:, link_i], kind="linear")(common_times)
        sim_y_at_t[:, link_i] = interp1d(sim_times, pos_y[:, link_i], kind="linear")(common_times)

    real_indices = np.searchsorted(real_times, common_times)
    real_indices = np.clip(real_indices, 0, len(real_times) - 1)

    model_x = np.zeros((n_times, _N_MODEL))
    model_y = np.zeros((n_times, _N_MODEL))
    for ti in range(n_times):
        model_x[ti], model_y[ti] = _spline_model_points(
            x_vals[real_indices[ti]], y_vals[real_indices[ti]],
        )

    # ── Rotate model points ──────────────────────────────────────────
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    model_x_rot = c * model_x - s * model_y
    model_y_rot = s * model_x + c * model_y

    # ── Translate: align heads at t=0 ────────────────────────────────
    dx = sim_x_at_t[0, 0] - model_x_rot[0, 0] * _MODEL_BL
    dy = sim_y_at_t[0, 0] - model_y_rot[0, 0] * _MODEL_BL
    model_x_al = model_x_rot * _MODEL_BL + dx
    model_y_al = model_y_rot * _MODEL_BL + dy

    # Shift to origin, m → mm
    sim_xp = (sim_x_at_t - sim_x_at_t[0, 0]) * 1000
    sim_yp = (sim_y_at_t - sim_y_at_t[0, 0]) * 1000
    mod_xp = (model_x_al - model_x_al[0, 0]) * 1000
    mod_yp = (model_y_al - model_y_al[0, 0]) * 1000

    # ── Centre of mass (geometric centroid of the plotted points, same
    #    method on both sides so the comparison stays consistent) ───────
    com_sim_yp = sim_yp.mean(axis=1); com_sim_xp = sim_xp.mean(axis=1)
    com_mod_yp = mod_yp.mean(axis=1); com_mod_xp = mod_xp.mean(axis=1)

    def _draw_com(ax, com_y, com_x):
        """Overlay the CoM path + time-coloured markers with a distinct cmap."""
        ax.plot(com_y, com_x, "-", color="0.5", linewidth=1.0, alpha=0.6, zorder=4)
        ax.scatter(com_y, com_x, c=common_times, cmap=cmap_com, norm=norm,
                   s=55, marker="*", edgecolors="k", linewidths=0.5, zorder=5)

    # ── Plot ─────────────────────────────────────────────────────────
    cmap = plt.cm.viridis
    cmap_com = plt.cm.autumn
    norm = Normalize(vmin=common_times[0], vmax=common_times[-1])

    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))

    ax = axes2[0]
    for ti in range(n_times):
        ax.plot(sim_yp[ti], sim_xp[ti], "o-", color=cmap(norm(common_times[ti])),
                markersize=5, linewidth=1.2, alpha=0.9)
    _draw_com(ax, com_sim_yp, com_sim_xp)
    ax.set_xlabel("Lateral (mm)"); ax.set_ylabel("Forward (mm)")
    ax.set_title(f"Simulation ({pos_x.shape[0]} steps, 16 links)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for ti in range(n_times):
        ax.plot(mod_yp[ti], mod_xp[ti], "s-", color=cmap(norm(common_times[ti])),
                markersize=4, linewidth=1.2, alpha=0.9)
    _draw_com(ax, com_mod_yp, com_mod_xp)
    ax.set_xlabel("Lateral (mm)"); ax.set_ylabel("Forward (mm)")
    ax.set_title(f"Experiment (aligned, {_N_MODEL} model pts)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = axes2[2]
    legend_handles = []
    for ti in range(n_times):
        color = cmap(norm(common_times[ti]))
        label = f"t={common_times[ti]:.2f}s" if ti % max(1, n_times // 6) == 0 else None
        ax.plot(sim_yp[ti], sim_xp[ti], "o-", color=color,
                markersize=6, linewidth=1.5, alpha=0.9)
        ax.plot(mod_yp[ti], mod_xp[ti], "^--", color=color,
                markersize=5, linewidth=1.0, alpha=0.7, markerfacecolor="none")
        if label:
            legend_handles.extend([
                Line2D([0], [0], marker="o", color=color, linestyle="-",
                       markersize=6, label=f"{label} sim"),
                Line2D([0], [0], marker="^", color=color, linestyle="--",
                       markersize=5, markerfacecolor="none", label=f"{label} exp"),
            ])
    # CoM paths on the overlay: filled star = sim, open star = exp
    ax.plot(com_sim_yp, com_sim_xp, "-", color="0.5", linewidth=1.0, alpha=0.6, zorder=4)
    ax.scatter(com_sim_yp, com_sim_xp, c=common_times, cmap=cmap_com, norm=norm,
               s=70, marker="*", edgecolors="k", linewidths=0.6, zorder=6)
    ax.plot(com_mod_yp, com_mod_xp, "--", color="0.5", linewidth=1.0, alpha=0.6, zorder=4)
    ax.scatter(com_mod_yp, com_mod_xp, c=common_times, cmap=cmap_com, norm=norm,
               s=70, marker="X", edgecolors="k", linewidths=0.6, zorder=6)
    legend_handles.extend([
        Line2D([0], [0], marker="*", color="0.4", linestyle="-", markersize=11,
               markerfacecolor="orange", label="CoM sim"),
        Line2D([0], [0], marker="X", color="0.4", linestyle="--", markersize=9,
               markerfacecolor="orange", label="CoM exp"),
    ])
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, ncol=2)
    ax.set_xlabel("Lateral (mm)"); ax.set_ylabel("Forward (mm)")
    ax.set_title("Overlay: Sim vs Exp (Aligned)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    fig2.suptitle(
        f"Body Shape Evolution: Sim vs Exp (ALIGNED)  —  {speed_label}\n"
        f"Rotated exp by {np.rad2deg(rot_angle):.1f}°, "
        f"translated by ({dx*1000:.1f}, {dy*1000:.1f}) mm",
        fontsize=11, y=1.03,
    )
    fig2.tight_layout(rect=[0, 0, 0.90, 1])
    cax_body = fig2.add_axes([0.925, 0.15, 0.010, 0.70])
    cax_com = fig2.add_axes([0.965, 0.15, 0.010, 0.70])
    fig2.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax_body)
    cax_body.set_title("body", fontsize=9)
    fig2.colorbar(ScalarMappable(norm=norm, cmap=cmap_com), cax=cax_com
                  ).set_label("Time (s)")
    cax_com.set_title("CoM", fontsize=9)

    if out_path:
        aligned_path = out_path.replace(".png", "_aligned_keypoints.png")
    else:
        aligned_path = os.path.join(sim_dir, f"comparison_{speed_label}_aligned_keypoints.png")
    fig2.savefig(aligned_path, dpi=150, bbox_inches="tight")
    print(f"Saved aligned keypoints plot to {aligned_path}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# COM positions 2D plot (third figure) — all iterations
# ---------------------------------------------------------------------------

def _plot_com_positions_2d(
    sim_dir: str,
    speed_type: str,
    link_array: np.ndarray,
    t_sim: np.ndarray,
    out_path: str | None = None,
    speed_label: str = "slow",
):
    """Plot COM (centre of mass) positions in 2D for real and sim at every iteration.

    Produces a single figure with:
      - Left panel:  sim COM trajectory (all steps, time-coloured)
      - Right panel: real COM trajectory (all steps, time-coloured)

    Both are shifted to start at the origin and expressed in body lengths (BL).
    """
    # ── Sim COM ──────────────────────────────────────────────────────
    com_x_sim = np.mean(link_array[:, :, sc.link_com_position_x], axis=1)
    com_y_sim = np.mean(link_array[:, :, sc.link_com_position_y], axis=1)
    # Shift to origin, convert to BL
    com_x_sim_bl = (com_x_sim - com_x_sim[0]) / _MODEL_BL
    com_y_sim_bl = (com_y_sim - com_y_sim[0]) / _MODEL_BL
    t_sim_bl = t_sim - t_sim[0]

    # ── Real COM ─────────────────────────────────────────────────────
    fname = _REAL_FILES[speed_type]["xy_bl"]
    csv_path = os.path.join(_KEYPOINTS_DIR, fname)
    df = pd.read_csv(csv_path)
    real_times = df["time_ms"].values.astype(float) / 1000.0
    x_cols = [c for c in df.columns if c.startswith("x")]
    y_cols = [c for c in df.columns if c.startswith("y")]
    com_x_real = df[x_cols].values.astype(float).mean(axis=1)
    com_y_real = df[y_cols].values.astype(float).mean(axis=1)
    # Shift to origin
    com_x_real_bl = com_x_real - com_x_real[0]
    com_y_real_bl = com_y_real - com_y_real[0]
    t_real_bl = real_times - real_times[0]

    # ── Common colour normalisation ──────────────────────────────────
    t_min = min(t_sim_bl[0], t_real_bl[0])
    t_max = max(t_sim_bl[-1], t_real_bl[-1])
    norm = Normalize(vmin=t_min, vmax=t_max)
    cmap = plt.cm.viridis

    # ── Plot ─────────────────────────────────────────────────────────
    fig, (ax_sim, ax_real, ax_overlay) = plt.subplots(1, 3, figsize=(21, 6.5))

    # --- Sim ---
    sc_sim = ax_sim.scatter(
        com_y_sim_bl, com_x_sim_bl, c=t_sim_bl, cmap=cmap, norm=norm,
        s=12, alpha=0.85, edgecolors="none", zorder=3,
    )
    ax_sim.plot(com_y_sim_bl, com_x_sim_bl, "-", color="0.4", linewidth=0.7, alpha=0.5, zorder=2)
    ax_sim.scatter(0, 0, marker="o", color="red", s=50, zorder=5, label="Start")
    ax_sim.set_xlabel("Lateral displacement [BL]")
    ax_sim.set_ylabel("Forward displacement [BL]")
    ax_sim.set_title(f"Sim COM ({link_array.shape[0]} steps)")
    ax_sim.set_aspect("equal")
    ax_sim.grid(True, alpha=0.3)
    ax_sim.legend(loc="upper left", fontsize=8)

    # --- Real ---
    sc_real = ax_real.scatter(
        com_y_real_bl, com_x_real_bl, c=t_real_bl, cmap=cmap, norm=norm,
        s=12, alpha=0.85, edgecolors="none", zorder=3,
    )
    ax_real.plot(com_y_real_bl, com_x_real_bl, "-", color="0.4", linewidth=0.7, alpha=0.5, zorder=2)
    ax_real.scatter(0, 0, marker="o", color="red", s=50, zorder=5, label="Start")
    ax_real.set_xlabel("Lateral displacement [BL]")
    ax_real.set_ylabel("Forward displacement [BL]")
    ax_real.set_title(f"Real COM ({len(real_times)} steps)")
    ax_real.set_aspect("equal")
    ax_real.grid(True, alpha=0.3)
    ax_real.legend(loc="upper left", fontsize=8)

    # --- Overlay ---
    ax_overlay.plot(com_y_sim_bl, com_x_sim_bl, "-", color="#2196F3",
                    linewidth=1.8, alpha=0.9, label="Sim COM", zorder=2)
    ax_overlay.plot(com_y_real_bl, com_x_real_bl, "-", color="black",
                    linewidth=1.8, alpha=0.9, label="Real COM", zorder=2)
    # Time-coloured scatter on overlay (thinned for readability)
    step_sim = max(1, len(t_sim_bl) // 300)
    step_real = max(1, len(t_real_bl) // 300)
    ax_overlay.scatter(
        com_y_sim_bl[::step_sim], com_x_sim_bl[::step_sim],
        c=t_sim_bl[::step_sim], cmap=cmap, norm=norm,
        s=18, alpha=0.9, edgecolors="none", zorder=3,
    )
    ax_overlay.scatter(
        com_y_real_bl[::step_real], com_x_real_bl[::step_real],
        c=t_real_bl[::step_real], cmap=cmap, norm=norm,
        s=18, alpha=0.9, edgecolors="none", marker="s", zorder=3,
    )
    ax_overlay.scatter(0, 0, marker="o", color="red", s=50, zorder=5, label="Start")
    ax_overlay.set_xlabel("Lateral displacement [BL]")
    ax_overlay.set_ylabel("Forward displacement [BL]")
    ax_overlay.set_title("Overlay: Sim vs Real COM")
    ax_overlay.set_aspect("equal")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"COM Positions 2D — All Iterations  —  {speed_label.capitalize()}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Single colourbar for all panels
    cax = fig.add_axes([0.94, 0.15, 0.012, 0.70])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Time [s]")

    # Save
    if out_path:
        com_path = out_path.replace(".png", "_com_positions_2d.png")
    else:
        com_path = os.path.join(sim_dir, f"comparison_{speed_label}_com_positions_2d.png")
    fig.savefig(com_path, dpi=150, bbox_inches="tight")
    print(f"Saved COM positions 2D plot to {com_path}")
    plt.close(fig)


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
