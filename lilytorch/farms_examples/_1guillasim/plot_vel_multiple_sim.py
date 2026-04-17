import h5py
import yaml
from farms_core.sensors.sensor_convention import sc
import numpy as np
import matplotlib.pyplot as plt
import os
from lilytorch.util.paths import save_path
from lilytorch.util.metrics import compute_speed_PCA

values=[0.5, 1, 1.5, 2]
stack_folder=os.path.join(save_path, "1guilla_swim_pd", "damping_2")
dirs = [
    os.path.join(stack_folder, dir2)
    for dir2 in os.listdir(stack_folder)
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for dir, value in zip(dirs, values):
    file = os.path.join(dir,'output', 'simulation.hdf5')
    animat_config_path0 = os.path.join(dir, "animat_config_0.yaml")

    with open(animat_config_path0, 'r') as f:
        animat_config0 = yaml.unsafe_load(f)

    freq =  animat_config0["extensions"][0]["config"]["freq"]


    with h5py.File(file, 'r') as f:

        link_array = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])
        times = np.array(f["times"])

        it_max = 8000

        links_vel = link_array[:,:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_y+1]
        links_pos = link_array[:,:,sc.link_com_position_x:sc.link_com_position_z+1]

        v_x, v_y = compute_speed_PCA(links_pos, links_vel)

        ax1.plot(times[:it_max], v_x[:it_max], label=str(freq)+'Hz')
        ax2.plot(times[:it_max], v_y[:it_max], label=str(freq)+'Hz')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('V_x [m/s]')
ax1.legend()
ax1.set_title('COM Velocity - Longitudinal')
ax1.grid(True)

ax2.set_xlabel('Time [s]')
ax2.set_ylabel('V_y [m/s]')
ax2.legend()
ax2.set_title('COM Velocity - Lateral')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join("figures", "com_velocities_freq.png"))
plt.close()


