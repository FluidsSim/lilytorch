
import h5py
from farms_core.sensors.sensor_convention import sc
import numpy as np
import matplotlib.pyplot as plt
import os

values=[0.5, 1, 1.5, 2]
dirs=[
    '/data/andreaferrario/ns_data/1guilla_experiments/0.5_20_12/',
    '/data/andreaferrario/ns_data/1guilla_experiments/1_20_12/',
    '/data/andreaferrario/ns_data/1guilla_experiments/1.5_20_12/',
    '/data/andreaferrario/ns_data/1guilla_experiments/2_20_12/'
]
tstops = [15, 9.5, 7, 6.1]

plt.figure()

for dir, tstop, value in zip(dirs, tstops, values):
    file = dir + 'simulation.hdf5'

    with h5py.File(file, 'r') as f:

        link_array = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])

        timestep = np.array(f["timestep"])
        n_iterations = link_array.shape[0]
        it_max = int(tstop / timestep)
        times = timestep * np.arange(n_iterations)

        v_links = link_array[:,:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_y+1]

        v_com =np.mean(v_links, axis=1)

        v_x = v_com[:it_max,0]
        v_y = v_com[:it_max,1]

        plt.plot(times[:it_max], v_x, label=str(value)+'Hz')
        # plt.plot(times[:it_max], v_y, label='v_y')

plt.xlabel('Time [s]')
plt.ylabel('V_x [m/s]')
plt.legend()
plt.title('COM Velocity')
plt.grid(True)
plt.savefig(os.path.join("figures", "com_velocities_freq.png"))
plt.close()




