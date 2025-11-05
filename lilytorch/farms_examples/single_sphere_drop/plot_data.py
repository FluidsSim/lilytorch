
import matplotlib.pyplot as plt

hdf5 = "/data/andreaferrario/ns_data/2025-11-04T11:11:33.996506/simulation.hdf5"

from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc

data = hdf5_to_dict(hdf5)

sphere=data["animats"][0]

sensor_array = sphere["sensors"]["links"]["array"][:,0,:]

com_vel = sensor_array[:,sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

plt.plot(com_vel)

plt.show()