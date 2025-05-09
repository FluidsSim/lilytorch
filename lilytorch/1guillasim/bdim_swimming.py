
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
from util.plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import farms_pylog as pylog
import numpy as np


def main(**kwargs):
    """
    Exercise example, running a single simulation and plotting the results
    """
    log_path = './logs/example_single/' # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations    = 100001,
        controller      = "sine",
        swimming_mode   = "bdim",
        amp             = np.ones(8),
        bias            = 0,
        timestep        = 0.001,
        twl             = 1,
        freq            = 2,
        log_path        = log_path,
        compute_metrics = 3,
        headless        = True,
        return_network  = True,
        gravity         = [0,0,-9.81],
        video_record    = False,
        video_name      = "swimming_example",
        spawn           = {
            'loader': 0,
            'mode': "TRANSVERSE",
            'pose': [0.0, 0.0, -0.3, 0.0, 0.0, 3.141592653589793],
            'velocity': [0.0, 0., 0.0, 0.0, 0.0, 0.0],
            'extras': {}
            },
        **kwargs
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )


if __name__ == '__main__':
    main()



