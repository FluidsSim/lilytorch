
from firing_rate_controller import FiringRateController
from util.mp_util import sweep_1d
from util.rw import save_object
from metrics import compute_controller
import os
from tqdm import tqdm
from scipy.integrate import ode

def pretty(d, indent=1):
   """
   print dictionary d in a pretty format
   """
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def step_rk(time, state, timestep, f):
    """Step"""
    k1 = timestep*f(time, state) 
    k2 = timestep*f(time + timestep / 2, state + k1 / 2) 
    k3 = timestep*f(time + timestep / 2, state + k2 / 2) 
    k4 = timestep*f(time + timestep, state + k3) 
    return state + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def step_ode(time, solver, timestep):
    """Step"""
    return solver.integrate(time+timestep)


def run_single(pars):
    """
    Run simulation
    Parameters
    ----------
    pars: <SimulationParameters class>
        class of simulation parameters
    Returns
    -------
    network: the FiringRateController() class
    """ 
    
    network = FiringRateController(
        pars
        )
    dt    = network.timestep
    times = network.times
    # Run network ODE

    solver = ode(f=network.f)
    solver.set_initial_value(y=network.state[0], t=0.0)
    solver.set_integrator(
            'vode', 
            atol=1e-3, 
            rtol=1e-3, 
            method="bdf",
            )

    _iterator = (
        tqdm(range(network.n_iterations-1))
        if network.pars.show_progress
        else range(network.n_iterations-1)
    )

    for i in _iterator:
        # network.state[i+1, :] = step_ode(times[i], solver, dt)
        network.state[i+1, :] = step_rk(times[i], network.state[i, :], dt, network.f)
    
    
    network.metrics=compute_controller(network)

    if pars.log_path!="":
        os.makedirs(pars.log_path, exist_ok=True)
        save_object(network, '{}controller{}'.format(pars.log_path, pars.simulation_i))
    if pars.compute_metrics:
        network.metrics=compute_controller(network)
        if pars.print_metrics:
            pretty(network.metrics)



    return network


def run_multiple(pars_list, num_process=6):
    """
    Run multiple simulation in parallel
    Parameters
    ----------
    pars_list: list of <SimulationParameters class>
        list of of simulation parameter classes
    Returns
    -------
    network_list: list of FiringRateController() classes
    """ 
    return sweep_1d(run_single, pars_list, num_process=num_process)








