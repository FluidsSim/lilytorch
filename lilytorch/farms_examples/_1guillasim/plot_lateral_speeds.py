
import numpy as np
import matplotlib.pyplot as plt
import os
from farms_core.experiment.data import ExperimentData
from farms_core.io.hdf5 import hdf5_to_dict
from lilytorch.util.paths import save_path
import random


# stack_folder=os.path.join(save_path, "2guilla","fb_off")
# stack_folder=os.path.join(save_path, "2guilla_2","fb_on")
stack_folder=os.path.join(save_path, "1guilla_swim_pd")
subdirs = [
    os.path.join(stack_folder, dir2)
    for dir2 in os.listdir(stack_folder)
]
random.shuffle(subdirs)


# subdirs = [os.path.join(save_path, "2026-02-04T15:50:32.829599")]

def plot_lateral_speeds(dir):
    ''' Plot lateral speeds from simulation data '''

    nn_file_path = os.path.join(dir, "output", "nn_data.hdf5")
    sim_file_path = os.path.join(dir, "output", "simulation.hdf5")

    nn_data                 = hdf5_to_dict(nn_file_path)
    exp_data                = ExperimentData.from_file(sim_file_path)

    for animat_i, animat in enumerate(exp_data.animats):

        times                   = exp_data.times
        com_positions           = np.mean(animat.sensors.links.com_positions(), axis=1)[:,:2]
        lateral_speeds          = nn_data['animats'][animat_i]['lateral_speed']
        lateral_speeds_filtered = nn_data['animats'][animat_i]['lateral_speed_filtered']

        plt.subplot(1,2,1)
        plt.plot(com_positions[:,0], com_positions[:,1], label=f'Animat {animat_i}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Center of Mass Trajectory')
        plt.legend()
        plt.grid()


        plt.subplot(1,2,2)
        plt.plot(times, lateral_speeds, label=f'Animat {animat_i} - Lateral Speed')
        plt.plot(times, lateral_speeds_filtered, label=f'Animat {animat_i} - Filtered', linestyle='--')
        plt.xlabel('Time step')
        plt.ylabel('Lateral Speed')
        plt.title('Lateral Speed over Time')
        plt.legend()
        plt.grid()

    plt.show()

for dir in subdirs:
    print(f"Processing directory: {dir}")
    plot_lateral_speeds(dir)
    plt.clf()