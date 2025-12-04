
import os
import matplotlib.pyplot as plt
import numpy as np
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc
from lilytorch.util.yaml_operations import yaml2pyobject

dir = "/data/andreaferrario/ns_data/2025-11-23T14:55:17.471975/"

uinf = -0.02501
D=0.005

data = hdf5_to_dict(dir + "simulation.hdf5")
times=data["times"][:-1]


sphere=data["animats"][0]

sensor_array = sphere["sensors"]["links"]["array"][:,0,:]

com_vel = sensor_array[:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]/uinf
ang_vel = D*sensor_array[:,sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1]/uinf


labels=["u_x","u_z","ang"]
cs=['r','g','b']
# for i in range(3):
#     plt.plot(com_vel[:,i], label=labels[i])
#     plt.plot(ang_vel[:,i], label=labels[i])

time_normalized = -times*uinf/D

plt.plot(time_normalized,com_vel[:,0], label=labels[0], color=cs[0])
plt.plot(time_normalized,com_vel[:,2], label=labels[1], color=cs[1])
plt.plot(time_normalized,ang_vel[:,1], label=labels[2], color=cs[2])
plt.legend()
plt.xlabel("t U_t/D")
plt.ylabel("Normalized COM velocity")
plt.title("Normalized cylinder velocity")




# vp = np.genfromtxt('data_to_save/vp.csv', delimiter=',')
# up = np.genfromtxt('data_to_save/up.csv', delimiter=',')
# wp = np.genfromtxt('data_to_save/wp.csv', delimiter=',')


# def convert_range(arr, old_min=0, old_max=1.2, new_min=-0.2, new_max=1.2):
#     scale = (new_max - new_min) / (old_max - old_min)
#     return new_min + (arr - old_min) * scale

# vp[:,1] = convert_range(vp[:,1])
# up[:,1] = convert_range(up[:,1])
# wp[:,1] = convert_range(wp[:,1])

# plt.scatter(up[:,0]-1., up[:,1], color=cs[1], s=6)
# plt.scatter(wp[:,0], wp[:,1], color=cs[2], s=3)
# plt.scatter(vp[:,0], vp[:,1], color=cs[0], s=3)

plt.ylim([-0.2,1.2])



plt.savefig("figures/sphere_com_velocity.png", dpi=300)

