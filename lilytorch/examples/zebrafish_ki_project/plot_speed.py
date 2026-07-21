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
import pandas as pd
import yaml
from farms_core.sensors.sensor_convention import sc
from scipy.signal import butter, filtfilt

from lilytorch.integration.kinematics import kinematics_interpolation
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


def _reconstruct_desired_positions(run_dir: str, n_joints: int, nt: int, timestep: float):
    """Reconstruct desired joint positions from the source xlsx kinematics file.

    Mirrors the interpolation logic in the PD controller and KinematicsController:
    load xlsx → trim time column if present → optional low-pass filter →
    optional zero-row prepend for startup stability → interpolate to the simulation
    timestep. Returns an (nt, n_joints) array, or None if the config / xlsx is
    unavailable.
    """
    cfg_path = os.path.join(run_dir, "animat_config_0.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = yaml.unsafe_load(f)
        ctrl_cfg = next(
            (e["config"] for e in cfg.get("extensions", [])
             if "pd_controller" in e.get("loader", "")),
            None,
        )
        if ctrl_cfg is None:
            return None
        data_folder = ctrl_cfg["data_folder"]
        file_path = ctrl_cfg.get("file_path")
        sampling = float(ctrl_cfg.get("kinematics_sampling", timestep))
        invert = bool(ctrl_cfg.get("kinematics_invert", True))
        lowpass_cutoff = ctrl_cfg.get("lowpass_cutoff")
        lowpass_order = int(ctrl_cfg.get("lowpass_order", 4))
    except Exception:
        return None

    if file_path is None:
        mode = ctrl_cfg.get("mode", "slow")
        fname = "joints_positions_slow.xlsx" if mode == "slow" else "joints_positions_fast.xlsx"
        xlsx_path = os.path.join(data_folder, fname)
    else:
        xlsx_path = file_path
        if not os.path.isabs(xlsx_path):
            xlsx_path = os.path.join(data_folder, xlsx_path)

    if not os.path.exists(xlsx_path):
        return None

    kin = pd.read_excel(xlsx_path).to_numpy(dtype=float)
    if kin.shape[1] == n_joints + 1:
        time_col = kin[:, 0]
        sampling = float(np.median(np.diff(time_col)))
        kin = kin[:, 1:]
    elif kin.shape[1] < n_joints:
        return None
    else:
        kin = kin[:, :n_joints]

    if lowpass_cutoff is not None and lowpass_cutoff > 0:
        fs = 1.0 / sampling
        nyq = 0.5 * fs
        b, a = butter(lowpass_order, lowpass_cutoff / nyq, btype="low")
        kin = np.column_stack([
            filtfilt(b, a, kin[:, j])
            for j in range(kin.shape[1])
        ])

    if invert:
        kin = -kin

    # Mirror the controller's startup prepend: a zero-row is inserted when the
    # first target frame is non-zero, so the first PD error starts from zero.
    raw_kin = kin
    if not np.allclose(raw_kin[0], 0.0):
        kin = np.vstack([np.zeros_like(raw_kin[0]), raw_kin])
        time_vector = np.arange(0, raw_kin.shape[0] * sampling, sampling) + sampling
        time_vector = np.insert(time_vector, 0, 0.0)
    else:
        kin = raw_kin
        time_vector = np.arange(0, raw_kin.shape[0] * sampling, sampling)

    # Add end-time padding to cover the full simulation.
    n_iter = nt - 1
    end_time = timestep * n_iter
    if end_time > 0:
        kin = np.insert(
            arr=kin,
            obj=kin.shape[0],
            values=np.repeat(a=[kin[-1, :]], repeats=int(end_time / sampling) + 1, axis=0),
            axis=0,
        )
        time_vector = np.insert(
            arr=time_vector,
            obj=time_vector.shape[0],
            values=np.linspace(
                time_vector[-1] + timestep,
                time_vector[-1] + end_time,
                int(end_time / sampling) + 1,
            ),
        )

    interp = kinematics_interpolation(
        kin_times=time_vector,
        kinematics=kin,
        timestep=timestep,
        n_iterations=n_iter,
    )
    return interp


def _load_joint_tracking(run_dir: str):
    """Return (times, pos_actual, pos_desired, vel_actual, joint_names).

    pos_actual:  (nt, n_joints) measured joint positions   [rad]
    pos_desired: (nt, n_joints) commanded joint positions  [rad]
    vel_actual:  (nt, n_joints) measured joint velocities  [rad/s]

    If sc.joint_cmd_position is not recorded (all zeros), desired positions are
    reconstructed from the source xlsx via _reconstruct_desired_positions().
    """
    hdf5_path = os.path.join(run_dir, "output", "simulation.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"simulation.hdf5 not found in {run_dir}/output/")

    with h5py.File(hdf5_path, "r") as f:
        joints_group = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]
        joints_array = np.array(joints_group["array"])
        joint_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in np.array(joints_group["names"])
        ]
        if "times" in f:
            times    = np.array(f["times"])[: joints_array.shape[0]]
            timestep = float(times[1] - times[0]) if len(times) > 1 else float(np.array(f["timestep"]))
        else:
            timestep = float(np.array(f["timestep"]))
            times    = timestep * np.arange(joints_array.shape[0])

    pos_actual  = joints_array[:, :, sc.joint_position]      # (nt, n_joints)
    pos_desired = joints_array[:, :, sc.joint_cmd_position]  # (nt, n_joints)
    vel_actual  = joints_array[:, :, sc.joint_velocity]      # (nt, n_joints)

    # Fallback: reconstruct desired positions from source xlsx when not stored
    if not np.any(pos_desired != 0):
        reconstructed = _reconstruct_desired_positions(
            run_dir, pos_actual.shape[1], joints_array.shape[0], timestep
        )
        if reconstructed is not None:
            pos_desired = reconstructed

    return times, pos_actual, pos_desired, vel_actual, joint_names


def _make_joint_figure(n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    return fig, np.asarray(axes).flatten()


def _finalise_joint_figure(fig, axes, n_joints, n_cols, n_rows, ylabel, title):
    for j in range(n_joints, len(axes)):
        axes[j].set_visible(False)
    for row in range(n_rows):
        axes[row * n_cols].set_ylabel(ylabel)
    last_row_start = (n_rows - 1) * n_cols
    for col in range(n_cols):
        idx_ax = last_row_start + col
        if idx_ax < len(axes):
            axes[idx_ax].set_xlabel("Time [s]")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()


def _plot_joint_grids(run_dirs, out_path):
    """Create joint-position, joint-velocity tracking, velocity consistency, and error figures.

    Saves:
        <stem>_joint_pos.png                    — desired (--) vs actual (—) positions per joint
        <stem>_joint_vel.png                    — dΘ_des/dt (--) vs actual (—) velocities per joint
        <stem>_joint_vel_consistency.png        — dΘ_act/dt (--) vs recorded vel (—) per joint
        <stem>_joint_pos_error.png              — position tracking error (actual − desired) per joint
        <stem>_joint_vel_error.png              — velocity tracking error (actual − dΘ_des/dt) per joint
        <stem>_joint_vel_consistency_error.png  — consistency error (recorded vel − dΘ_act/dt) per joint
    """
    tracking_data = []
    joint_names_global = None

    for run_dir in run_dirs:
        try:
            t, pa, pd, va, jnames = _load_joint_tracking(run_dir)
        except FileNotFoundError as exc:
            print(f"[skip joints] {exc}")
            tracking_data.append(None)
            continue
        tracking_data.append((t, pa, pd, va))
        if joint_names_global is None:
            joint_names_global = jnames

    if joint_names_global is None:
        return  # nothing to plot

    n_joints = len(joint_names_global)
    n_cols = 5
    n_rows = (n_joints + n_cols - 1) // n_cols

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(run_dirs), 1)))

    fig_pos,  axes_pos  = _make_joint_figure(n_rows, n_cols)
    fig_vel,  axes_vel  = _make_joint_figure(n_rows, n_cols)
    fig_con,  axes_con  = _make_joint_figure(n_rows, n_cols)
    fig_epos, axes_epos = _make_joint_figure(n_rows, n_cols)
    fig_evel, axes_evel = _make_joint_figure(n_rows, n_cols)
    fig_econ, axes_econ = _make_joint_figure(n_rows, n_cols)

    for idx, (run_dir, data) in enumerate(zip(run_dirs, tracking_data)):
        if data is None:
            continue
        times_j, pos_act, pos_des, vel_act = data
        color = colors[idx]
        label = _label_for_dir(run_dir)
        dt = times_j[1] - times_j[0] if len(times_j) > 1 else 1e-3
        vel_des       = np.diff(pos_des, axis=0) / dt  # (nt-1, n_joints)
        vel_from_pos  = np.diff(pos_act, axis=0) / dt  # (nt-1, n_joints)
        pos_err       = pos_act - pos_des                # (nt,   n_joints)
        vel_err       = vel_act[1:] - vel_des            # (nt-1, n_joints)
        vel_con_err   = vel_act[1:] - vel_from_pos       # (nt-1, n_joints)

        for j in range(n_joints):
            leg = j == 0
            jname = joint_names_global[j]

            axes_pos[j].plot(
                times_j, np.degrees(pos_act[:, j]),
                color=color, label=(f"{label} actual" if leg else "_nolegend_"),
            )
            axes_pos[j].plot(
                times_j, np.degrees(pos_des[:, j]),
                "--", color=color, alpha=0.7,
                label=(f"{label} desired" if leg else "_nolegend_"),
            )
            axes_pos[j].set_title(jname, fontsize=9)

            axes_vel[j].plot(
                times_j[1:], np.degrees(vel_act[1:, j]),
                color=color, label=(f"{label} actual" if leg else "_nolegend_"),
            )
            axes_vel[j].plot(
                times_j[1:], np.degrees(vel_des[:, j]),
                "--", color=color, alpha=0.7,
                label=(f"{label} dΘ_des/dt" if leg else "_nolegend_"),
            )
            axes_vel[j].set_title(jname, fontsize=9)

            axes_con[j].plot(
                times_j[1:], np.degrees(vel_act[1:, j]),
                color=color, label=(f"{label} recorded" if leg else "_nolegend_"),
            )
            axes_con[j].plot(
                times_j[1:], np.degrees(vel_from_pos[:, j]),
                "--", color=color, alpha=0.7,
                label=(f"{label} dΘ_act/dt" if leg else "_nolegend_"),
            )
            axes_con[j].set_title(jname, fontsize=9)

            axes_epos[j].plot(
                times_j, np.degrees(pos_err[:, j]),
                color=color, label=(label if leg else "_nolegend_"),
            )
            axes_epos[j].axhline(0, color="k", linewidth=0.6, linestyle=":")
            axes_epos[j].set_title(jname, fontsize=9)

            axes_evel[j].plot(
                times_j[1:], np.degrees(vel_err[:, j]),
                color=color, label=(label if leg else "_nolegend_"),
            )
            axes_evel[j].axhline(0, color="k", linewidth=0.6, linestyle=":")
            axes_evel[j].set_title(jname, fontsize=9)

            axes_econ[j].plot(
                times_j[1:], np.degrees(vel_con_err[:, j]),
                color=color, label=(label if leg else "_nolegend_"),
            )
            axes_econ[j].axhline(0, color="k", linewidth=0.6, linestyle=":")
            axes_econ[j].set_title(jname, fontsize=9)

    _finalise_joint_figure(
        fig_pos, axes_pos, n_joints, n_cols, n_rows,
        "Position [deg]", "Joint position tracking — desired (--) vs actual (—)",
    )
    _finalise_joint_figure(
        fig_vel, axes_vel, n_joints, n_cols, n_rows,
        "Velocity [deg/s]", "Joint velocity tracking — dΘ_des/dt (--) vs actual (—)",
    )
    _finalise_joint_figure(
        fig_con, axes_con, n_joints, n_cols, n_rows,
        "Velocity [deg/s]",
        "Joint velocity consistency — recorded (—) vs dΘ_act/dt (--)",
    )
    _finalise_joint_figure(
        fig_epos, axes_epos, n_joints, n_cols, n_rows,
        "Error [deg]", "Joint position tracking error (actual − desired)",
    )
    _finalise_joint_figure(
        fig_evel, axes_evel, n_joints, n_cols, n_rows,
        "Error [deg/s]", "Joint velocity tracking error (actual − dΘ_des/dt)",
    )
    _finalise_joint_figure(
        fig_econ, axes_econ, n_joints, n_cols, n_rows,
        "Error [deg/s]", "Joint velocity consistency error (recorded − dΘ_act/dt)",
    )

    base = out_path[:-4] if out_path.endswith(".png") else out_path
    for fig, suffix in [
        (fig_pos,  "_joint_pos.png"),
        (fig_vel,  "_joint_vel.png"),
        (fig_con,  "_joint_vel_consistency.png"),
        (fig_epos, "_joint_pos_error.png"),
        (fig_evel, "_joint_vel_error.png"),
        (fig_econ, "_joint_vel_consistency_error.png"),
    ]:
        path = base + suffix
        fig.savefig(path)
        print(f"Saved: {path}")
        plt.close(fig)


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
        default=0.017,
        help="Body length in metres for normalisation (default: 0.017 m).",
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

    # ── Joint tracking figures ─────────────────────────────────────────────
    _plot_joint_grids(run_dirs, out_path)


if __name__ == "__main__":
    main()
