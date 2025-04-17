

from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
from util.rw import load_object
import matplotlib.pyplot as plt
from metrics import *


dir           = "/data/andreaferrario/ns_data/2025-04-15T15:10:48.835164/"
controller_id = 0

controller = load_object(dir+"controller"+str(controller_id))

metrics = {}
sim_fraction = 1
exit_it=10000#controller.exit_iteration
print(exit_it)


times= controller.times[:exit_it]
links_positions       = controller.links_positions[:exit_it]
link_velocities       = controller.links_velocities[:exit_it]
joints_active_torques = controller.joints_active_torques[:exit_it]


n_steps            = links_positions.shape[0]
n_steps_considered = round(n_steps * sim_fraction)

times = times[-n_steps_considered:exit_it]

# ------ COMPUTE FORWARD AND LATERAL SPEEDS ------
(fspeed, lspeed) = compute_speed_PCA(
    links_positions,
    link_velocities,
    sim_fraction = sim_fraction
    )
fspeed = fspeed[:exit_it]
lspeed = lspeed[:exit_it]

plt.figure("fpeed PCA")
plt.plot(times, fspeed, label="Forward speed")

plt.figure("lspeed PCA")
plt.plot(times, lspeed, label="l speed")


plt.show()




