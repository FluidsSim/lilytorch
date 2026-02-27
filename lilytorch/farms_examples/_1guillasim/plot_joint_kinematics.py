
import glob
import os
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from farms_core.sensors.sensor_convention import sc
from farms_core.io.sdf import ModelSDF

matplotlib.rcParams.update({'font.size': 12})


def load_joint_data(sim_dir, animat_id=0):
    ''' Load joint kinematics data from a simulation directory.

    Parameters
    ----------
    sim_dir : str
        Path to the simulation directory (e.g.
        /data/andreaferrario/ns_data/2026-02-27T12:31:04.108378).
        Expects ``output/simulation.hdf5`` inside.
    animat_id : int
        Index of the animat to load.
    '''

    file_path = os.path.join(sim_dir, "output", "simulation.hdf5")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Simulation file not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        times         = np.array(f["times"])
        joints_array  = np.array(
            f["FARMSLISTanimats"][str(animat_id)]["sensors"]["joints"]["array"]
        )

    joints_pos = joints_array[:, :, sc.joint_position]
    joints_vel = joints_array[:, :, sc.joint_velocity]

    return times, joints_pos, joints_vel


def load_desired_kinematics(csv_path):
    ''' Load desired joint kinematics from a CSV file.

    First column is time, remaining columns are joint positions.
    '''

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    times_des    = data[:, 0]
    joints_des   = data[:, 1:]
    return times_des, joints_des


def detect_animats(sim_dir):
    ''' Return sorted list of animat indices found in the directory. '''

    pattern = os.path.join(sim_dir, "animat_config_*.yaml")
    files   = sorted(glob.glob(pattern))
    indices = [
        int(os.path.basename(f).replace("animat_config_", "").replace(".yaml", ""))
        for f in files
    ]
    return indices


def load_joint_names(sim_dir, animat_id=0):
    ''' Load joint names from the SDF file referenced in the animat config. '''

    config_path = os.path.join(sim_dir, f"animat_config_{animat_id}.yaml")
    with open(config_path, 'r') as f:
        animat_config = yaml.unsafe_load(f)

    sdf_file  = animat_config['sdf']
    model_sdf = ModelSDF.read(sdf_file)[0]
    joint_names = [joint.name for joint in model_sdf.joints if joint.type != 'fixed']
    return joint_names


def _plot_joint_group(
    times_sim, joints_pos_group, joints_des_interp_group,
    names_group, group_label, animat_id=0, n_cols=3, save_path=None,
):
    ''' Plot desired vs actual for a subset of joints. '''

    n_joints = len(names_group)
    if n_joints == 0:
        return

    n_rows = int(np.ceil(n_joints / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True)
    fig.suptitle(f'{group_label} — Animat {animat_id}', fontsize=16)
    axes = np.atleast_2d(axes)

    for idx in range(n_joints):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        # ax.plot(times_sim, joints_des_interp_group[:, idx], label='Desired', linestyle='--', alpha=0.8)
        ax.plot(times_sim, joints_pos_group[:, idx],         label='Actual',  alpha=0.8)
        ax.set_title(names_group[idx])
        ax.set_ylabel('Position [rad]')
        ax.grid(True)
        if idx == 0:
            ax.legend(fontsize=9)

    for col in range(n_cols):
        axes[-1, col].set_xlabel('Time [s]')

    for idx in range(n_joints, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        tag = group_label.lower().replace(' ', '_')
        base, ext = os.path.splitext(save_path)
        out_path = f"{base}_{tag}{ext}"
        plt.savefig(out_path, dpi=150)
        print(f"Figure saved to {out_path}")
    else:
        plt.show()

    plt.close()


def plot_desired_vs_actual(
    times_sim, joints_pos, times_des, joints_des,
    animat_id=0, joint_names=None, save_path=None,
):
    ''' Plot desired vs actual joint positions in separate figures for
    spine (joint_body_*) and limb (joint_leg_*) joints. '''

    n_joints_sim = joints_pos.shape[1]
    n_joints_des = joints_des.shape[1]
    n_joints     = min(n_joints_sim, n_joints_des)

    if joint_names is None:
        joint_names = [f'Joint {j}' for j in range(n_joints)]

    # Interpolate desired kinematics onto the simulation time grid
    joints_des_interp = np.zeros((len(times_sim), n_joints))
    for j in range(n_joints):
        joints_des_interp[:, j] = np.interp(times_sim, times_des, joints_des[:, j])

    # Split indices into spine and limb groups
    spine_idx = [j for j in range(n_joints) if 'leg' not in joint_names[j]]
    limb_idx  = [j for j in range(n_joints) if 'leg' in joint_names[j]]

    # Spine joints
    _plot_joint_group(
        times_sim,
        joints_pos[:, spine_idx],
        joints_des_interp[:, spine_idx],
        [joint_names[j] for j in spine_idx],
        'Spine Joints',
        animat_id=animat_id,
        save_path=save_path,
    )

    # Limb joints
    _plot_joint_group(
        times_sim,
        joints_pos[:, limb_idx],
        joints_des_interp[:, limb_idx],
        [joint_names[j] for j in limb_idx],
        'Limb Joints',
        animat_id=animat_id,
        save_path=save_path,
    )


if __name__ == "__main__":

    # ---- User settings ----
    sim_dir   = "/data/andreaferrario/ns_data/2026-02-27T12:31:04.108378"
    animat_id = None   # Set to an int to plot a specific animat, or None to plot all
    save_path = None   # Set to a file path to save the figure, or None to show interactively
    # -----------------------

    # Desired kinematics CSV (same file used by the PD controller)
    base_dir  = os.path.dirname(__file__)
    csv_path  = os.path.join(
        base_dir, '..', 'pleurodeles', 'salamander_kinematics_2D_x15.csv'
    )
    times_des, joints_des = load_desired_kinematics(csv_path)

    if animat_id is not None:
        animat_ids = [animat_id]
    else:
        animat_ids = detect_animats(sim_dir)
        if not animat_ids:
            animat_ids = [0]
        print(f"Detected animats: {animat_ids}")

    for aid in animat_ids:
        times_sim, joints_pos, joints_vel = load_joint_data(sim_dir, animat_id=aid)
        joint_names = load_joint_names(sim_dir, animat_id=aid)
        plot_desired_vs_actual(
            times_sim, joints_pos, times_des, joints_des,
            animat_id=aid,
            joint_names=joint_names,
            save_path=save_path,
        )
