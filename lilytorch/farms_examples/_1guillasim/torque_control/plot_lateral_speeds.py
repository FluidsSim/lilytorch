
import numpy as np
import matplotlib.pyplot as plt
import os
from farms_core.experiment.data import ExperimentData
from farms_core.io.hdf5 import hdf5_to_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
nn_file_path = os.path.join(current_dir, "1guilla_test_configs/output", "nn_data.hdf5")
sim_file_path = os.path.join(current_dir, "1guilla_test_configs/output", "simulation.hdf5")

nn_data                 = hdf5_to_dict(nn_file_path)
simulation_data         = hdf5_to_dict(sim_file_path)
exp_data                = ExperimentData.from_file(sim_file_path)
times                   = exp_data.times
com_positions           = np.mean(exp_data.animats[0].sensors.links.com_positions(), axis=1)[:,:2]
lateral_speeds          = nn_data['animats'][0]['lateral_speed']
lateral_speeds_filtered = nn_data['animats'][0]['lateral_speed_filtered']



plt.subplot(1,2,1)
plt.plot(com_positions[:,0], com_positions[:,1])
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Center of Mass Trajectory')
plt.grid()


plt.subplot(1,2,2)
plt.plot(times, lateral_speeds, label='Lateral Speed')
plt.plot(times, lateral_speeds_filtered, label='Filtered Lateral Speed')
plt.xlabel('Time step')
plt.ylabel('Lateral Speed')
plt.title('Lateral Speed over Time')
plt.legend()
plt.grid()
plt.show()