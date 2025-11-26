
import os
import matplotlib.pyplot as plt
import numpy as np
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc
from lilytorch.util.yaml_operations import yaml2pyobject
import matplotlib
matplotlib.rc('font', **{"size":20})

dir = "/data/andreaferrario/ns_data/2025-11-19T10:37:54.990651/"
format=".svg"

data = hdf5_to_dict(dir + "simulation.hdf5")
timestep = data["times"][1] - data["times"][0]
times=data["times"][:-1]


sphere=data["animats"][0]

sensor_array = sphere["sensors"]["links"]["array"][:,0,:]

com_vel = sensor_array[:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]
ang_vel = sensor_array[:,sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1]

urdf_pos = sensor_array[:,sc.link_urdf_position_x:sc.link_urdf_position_z+1]



labels=["u_x","u_z","ang"]
cs=['b','g','r']
# for i in range(3):
#     plt.plot(com_vel[:,i], label=labels[i])
#     plt.plot(ang_vel[:,i], label=labels[i])


plt.figure()
plt.plot(times,com_vel[:,2], label=labels[1], color=cs[1])
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Z-Velocity")
plt.savefig("figures/sphere_com_velocity_z."+format, dpi=300)

plt.figure()
plt.plot(times,com_vel[:,0], label=labels[0], color=cs[0])
plt.legend()
plt.xlabel("Time step")
plt.ylabel("X-Velocity")
plt.savefig("figures/sphere_com_velocity_x."+format, dpi=300)

plt.figure()
plt.plot(times,ang_vel[:,1], label=labels[2], color=cs[2])
plt.legend()
plt.xlabel("Time step")
plt.ylabel("w-Velocity")
plt.savefig("figures/sphere_com_angular_velocity_y."+format, dpi=300)

plt.figure()
plt.plot(times, urdf_pos[:,2], label="urdf_z", color='k')
plt.legend()
plt.xlabel("Time step")
plt.ylabel("URDF Z Position")
plt.title("Sphere URDF Z Position over Time")
plt.savefig("figures/sphere_urdf_z_position."+format, dpi=300)


timestamp = 0.1
iteration = int(timestamp / timestep)
zstamp = urdf_pos[iteration,2]
v = np.load(dir + "uv_field/v_" + str(iteration) + ".npy")
ygrid = np.load(dir + "uv_field/y_grid.npy")
xgrid = np.load(dir + "uv_field/x_grid.npy")
closest_idx = np.abs(ygrid - zstamp).argmin()


plt.figure()
plt.plot(xgrid, v[:, closest_idx], label="v at t=%.3f s" % timestamp)
plt.xlabel("X Position")
plt.ylabel("Velocity")
plt.title("Z-Velocity Profile at t=%.1f s" % timestamp)
plt.savefig("figures/sphere_velocity_profile_t%.2f_z%.3f." % (timestamp, zstamp) + format, dpi=300)

# from IPython import embed; embed()


plt.show()



# plt.savefig("sphere_com_velocity.png", dpi=300)

