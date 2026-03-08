import h5py
import yaml
from farms_core.sensors.sensor_convention import sc
import numpy as np
import matplotlib.pyplot as plt
import os
from lilytorch.util.paths import save_path
from lilytorch.util.metrics import compute_speed_PCA

BL=0.018

dir = os.path.join(save_path, "2026-02-24T14:32:20.397323")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

file = os.path.join(dir,'output', 'simulation.hdf5')
animat_config_path0 = os.path.join(dir, "animat_config_0.yaml")

with open(animat_config_path0, 'r') as f:
    animat_config0 = yaml.unsafe_load(f)

freq = animat_config0["extensions"][0]["config"]["freq"]
amp  = animat_config0["extensions"][0]["config"]["amp"]
twl  = animat_config0["extensions"][0]["config"]["twl"]


with h5py.File(file, 'r') as f:

    link_array = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
    times = np.array(f["times"])

    it_max = 1200

    links_vel = link_array[:,:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_y+1]
    links_pos = link_array[:,:,sc.link_com_position_x:sc.link_com_position_z+1]

    v_x, v_y = compute_speed_PCA(links_pos, links_vel)
    v_x=v_x/BL
    v_y=v_y/BL

    ax1.plot(times[:it_max], v_x[:it_max], label=str(freq)+'Hz, '+str(amp)+'deg, '+str(twl)+'twl',)
    ax2.plot(times[:it_max], v_y[:it_max], label=str(freq)+'Hz, '+str(amp)+'deg, '+str(twl)+'twl',)

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('V_x [BL/s]')
    ax1.legend()
    ax1.set_title('COM Velocity - Longitudinal')
    ax1.grid(True)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('V_y [BL/s]')
    ax2.legend()
    ax2.set_title('COM Velocity - Lateral')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join("figures", "com_velocities_zebrafish.png"))
    plt.close()

    plt.show()


