
import os
import matplotlib.pyplot as plt
import numpy as np
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc
from lilytorch.util.yaml_operations import yaml2pyobject

dir = "/data/andreaferrario/ns_data/2025-11-10T09:06:41.481791/"

uinf = -0.02501
D=0.005

data = hdf5_to_dict(dir + "simulation.hdf5")

sphere=data["animats"][0]

sensor_array = sphere["sensors"]["links"]["array"][:,0,:]

com_vel = sensor_array[:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]/uinf
ang_vel = D*sensor_array[:,sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1]/uinf


labels=["u_x","u_z","ang"]
for i in range(3):
    plt.plot(com_vel[:,i], label=labels[i])
    plt.plot(ang_vel[:,i], label=labels[i])

plt.legend()
plt.xlabel("Time step")
plt.ylabel("Normalized COM velocity")
plt.title("Sphere COM Velocity over Time")

plt.show()


# plt.savefig("sphere_com_velocity.png", dpi=300)




# name        = "video"
# img_name    = "curl"
# format      = ".mp4"
# dt          = 0.1
# slow_factor = 1
# tstop       = 2000
# video_name  = dir+name+format

# def vorticity(u, v, h):
#     """
#     Compute the vorticity of u,v in 2d - dv/dx-du/dy
#     """
#     dvdx = np.zeros_like(u)
#     dudy = np.zeros_like(u)
#     dvdx[1:-1, 1:-1] = (v[1:-1, 1:-1]-v[:-2, 1:-1])/h
#     dudy[1:-1, 1:-1] = (u[1:-1, 1:-1]-u[1:-1, :-2])/h
#     return dvdx-dudy


# uv_folder  = dir+"uv_field/"
# n          = 140500
# save_every = 500
# dt         = 0.0001
# h          = 0.00015625
# Nx         = 256
# Ny         = 2048
# xmin       = -0.02
# ymin       = 0.0
# xmax       = 0.02
# ymax       = 0.32
# h          = (xmax - xmin) / Nx
# vmax       = 5

# plt.figure(figsize=(2,12))
# for i in range(int(n/save_every)):
#     u_field = np.load(uv_folder+"u_"+str(i*save_every)+".npy")
#     v_field = np.load(uv_folder+"v_"+str(i*save_every)+".npy")
#     vort_field = vorticity(u_field, v_field, h)

#     plt.imshow(vort_field.T, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
#     # plt.colorbar(label='Vorticity')
#     # plt.title('Vorticity Field at t={:.2f}s'.format(i*save_every*dt))
#     # plt.xlabel('x (m)')
#     # plt.ylabel('y (m)')
#     # plt.savefig(uv_folder+"vorticity_"+str(i*save_every)+".png")
#     plt.close()




# uv_field=np.load()
# uv_field =
# u = uv_field["u"]
# v = uv_field["v"]










# plt.figure()
# plt.imshow(vorticity[0], cmap='RdBu', origin='lower')
# plt.colorbar(label='Vorticity')
# plt.title('Vorticity Field at t=0')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig("vorticity_field.png", dpi=300)




# plt.show()