
import h5py
from farms_core.sensors.sensor_convention import sc
import numpy as np
import matplotlib.pyplot as plt
import os

dir='/data/andreaferrario/ns_data/1guilla_experiments/2025-12-09T15:35:05.118333/'
file = dir + 'simulation.hdf5'


with h5py.File(file, 'r') as f:

    link_array = np.array(f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"])

    timestep = np.array(f["timestep"])
    n_iterations = link_array.shape[0]
    it_max = 95000
    times = timestep * np.arange(n_iterations)

    v_links = link_array[:,:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_y+1]

    v_com =np.mean(v_links, axis=1)

    v_x = v_com[:it_max,0]
    v_y = v_com[:it_max,1]


    plt.figure()
    plt.plot(times[:it_max], v_x, label='v_x')
    plt.plot(times[:it_max], v_y, label='v_y')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.title('COM Velocity')
    plt.grid(True)
    plt.savefig(os.path.join(dir, "com_velocity_plot.png"))
    plt.close()




