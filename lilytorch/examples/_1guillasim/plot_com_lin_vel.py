import h5py
import yaml
from farms_core.sensors.sensor_convention import sc
import numpy as np
import matplotlib.pyplot as plt
import os
from lilytorch.util.paths import save_path

dir = os.path.join(save_path, "2026-03-23T16:32:00.364793")


file = os.path.join(dir, 'output', 'simulation.hdf5')
animat_config_path0 = os.path.join(dir, "animat_config_0.yaml")

with open(animat_config_path0, 'r') as f:
    animat_config0 = yaml.unsafe_load(f)

freq = animat_config0["extensions"][0]["config"]["freq"]

with h5py.File(file, 'r') as f:

    link_array = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
    times = np.array(f["times"])

    it_max = 8000

    vel_x = link_array[:it_max, :, sc.link_com_velocity_lin_x]
    vel_z = link_array[:it_max, :, sc.link_com_velocity_lin_z]
    ang_vel_y = link_array[:it_max, :, sc.link_com_velocity_ang_y]
    n_links = vel_x.shape[1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

    for i in range(2):
        ax1.plot(times[:it_max], vel_x[:, i], label=f'link {i}')
        ax2.plot(times[:it_max], vel_z[:, i], label=f'link {i}')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('V_x [m/s]')
    ax1.legend(fontsize='small', ncol=2)
    ax1.set_title(f'COM Linear Velocity X - {freq}Hz')
    ax1.grid(True)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('V_z [m/s]')
    ax2.legend(fontsize='small', ncol=2)
    ax2.set_title(f'COM Linear Velocity Z - {freq}Hz')
    ax2.grid(True)

    omega = ang_vel_y[:, 0]

    # Pivot position (joint frame origin)
    pivot_x = link_array[:it_max, 0, sc.link_urdf_position_x]
    pivot_z = link_array[:it_max, 0, sc.link_urdf_position_z]

    # COM position
    com_x = link_array[:it_max, 0, sc.link_com_position_x]
    com_z = link_array[:it_max, 0, sc.link_com_position_z]

    # R = distance from pivot to COM
    R = np.sqrt((com_x - pivot_x)**2 + (com_z - pivot_z)**2)

    # Finite-difference velocity from COM positions
    dt = times[1] - times[0]
    vx_fd = np.gradient(com_x, dt)
    vz_fd = np.gradient(com_z, dt)
    V_lin = np.sqrt(vx_fd**2 + vz_fd**2)

    V_lin_true = np.sqrt(vel_x[:, 0]**2 + vel_z[:, 0]**2)


    ax3.plot(times[:it_max], np.abs(omega * R), label='|ω R| (link 0)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('|ω R| [m/s]')
    ax3.legend(fontsize='small')
    ax3.set_title(f'|ω R| (link 0) - {freq}Hz')
    ax3.grid(True)

    ax3.plot(times[:it_max], V_lin, 'r', label='√(Vx² + Vz²) (link 0)')
    ax3.plot(times[:it_max], V_lin_true, 'g', label='√(Vx² + Vz²) true (link 0)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('|V| [m/s]')
    ax3.legend(fontsize='small')
    ax3.set_title(f'Linear Speed (link 0) - {freq}Hz')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"com_lin_vel_{freq}Hz.png"))
    plt.close()
