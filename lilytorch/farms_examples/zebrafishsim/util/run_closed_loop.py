
from util.simulation_control import run_experiment
from util.mp_util import sweep_1d, sweep_2d
import os
from util.rw import save_object
from metrics import compute_all, compute_controller, compute_mechanical
import numpy as np

def pretty(d, indent=1):
   """
   print dictionary d in a pretty format
   """
   print("The computed metrics are \t")
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def run_single(pars):

    animat_data, network = run_experiment(pars)

    # store amimat data vars in network as numpy arrays and delete animat
    network.joints                = np.array(animat_data.sensors.joints.array)
    network.links_positions       = np.array(animat_data.sensors.links.urdf_positions())
    network.links_orientations    = np.array(animat_data.sensors.links.urdf_orientations())
    network.links_velocities      = np.array(animat_data.sensors.links.com_lin_velocities())
    network.joints_active_torques = np.array(animat_data.sensors.joints.active_torques())
    network.joints_velocities     = np.array(animat_data.sensors.joints.velocities_all())
    network.joints_positions      = np.array(animat_data.sensors.joints.positions_all())

    del animat_data

    if pars.compute_metrics==1:
        network.metrics=compute_controller(network)
    elif pars.compute_metrics==2:
        network.metrics=compute_mechanical(network)
    elif pars.compute_metrics==3:
        network.metrics=compute_all(network)
    else:
        network.metrics=None

    if pars.compute_metrics>0 and pars.print_metrics:
        pretty(network.metrics)

    if pars.log_path!="":
        os.makedirs(pars.log_path, exist_ok=True)
        save_object(network, '{}controller{}'.format(pars.log_path, pars.simulation_i))

    if pars.return_network==True:
        return network
    else:
        return None

def run_multiple(pars_list, num_process=6):

    return sweep_1d(run_single, pars_list, num_process=num_process)


def run_multiple2d(pars_list1, pars_list2, num_process=6):

    return sweep_2d(run_single, pars_list1, pars_list2, num_process=num_process)



