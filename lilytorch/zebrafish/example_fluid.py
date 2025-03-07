
from util.run_fluid_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import farms_pylog as pylog


def exercise_single(**kwargs):
    """
    Exercise example, running a single simulation and plotting the results
    """
    log_path = './logs/example_single/' # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        controller      = "sine",
        swimming_mode   = "bdim",
        amp             = 0.5,
        wavefrequency   = 1,
        freq            = 1,
        n_iterations    = 80001,
        timestep        = 0.001,
        log_path        = log_path,
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



