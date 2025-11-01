
from util.run_fluid_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import farms_pylog as pylog
import numpy as np

def exercise_single(**kwargs):
    """
    Exercise example, running a single simulation and plotting the results
    """
    amps=np.ones(15)*1
    # amps[:5]=0
    bias=0.*np.ones(15)
    # bias[:5]=0

    all_pars = SimulationParameters(
        controller      = "sine",
        swimming_mode   = "bdim",
        amp             = amps,
        bias            = bias,
        wavefrequency   = 1,
        freq            = 5,
        n_iterations    = 80000,
        timestep        = 0.001,
        compute_metrics = 3,
        n_joints        = 15,
        return_network  = True,
        headless        = True,
        **kwargs
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )



if __name__ == '__main__':
    exercise_single()
    plt.show()



