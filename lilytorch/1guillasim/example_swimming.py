
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

    amps = np.ones(8)*1
    # amps[6:]= 0.0

    all_pars = SimulationParameters(
        n_iterations    = 15001,
        n_joints        = 8,
        controller      = "sine",
        swimming_mode   = "drag",
        amp             = amps,
        bias            = 0,
        timestep        = 0.001,
        twl             = 0.8,
        freq            = 2,
        log_path        = log_path,
        compute_metrics = 3,
        return_network  = True,
        gravity         = [0,0,-9.81],
        video_record    = True,
        video_name      = "swimming_example",
        spawn           = {
            'loader': 0,
            'mode': "TRANSVERSE",
            'pose': [0.0, 0.0, -0., 0.0, 0.0, 3.141592653589793],
            'velocity': [0.0, 0., 0.0, 0.0, 0.0, 0.0],
            'extras': {}
            },
        **kwargs
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )


    pylog.info("Plotting the result")

    left_idx = controller.muscle_l
    right_idx = controller.muscle_r

    # example plot using plot_left_right
    plt.figure('muscle_activities_single')
    plot_left_right(controller.times, controller.state, left_idx, right_idx, cm="green", offset=0.1)
    plt.savefig(log_path+"muscle_activities.png", dpi=300, bbox_inches='tight')

    # example plot using plot_trajectory
    plt.figure("trajectory_single")
    plot_trajectory(controller)
    plt.savefig(log_path+"trajectory.png", dpi=300, bbox_inches='tight')

    # example plot using plot_time_histories_multiple_windows
    plt.figure("joint positions_single")
    plot_time_histories_multiple_windows(
        controller.times,
        controller.joints_positions,
        offset=-0.4,
        colors="green",
        ylabel="joint positions",
        savepath=log_path+"joint_positions.png",
        lw=1
        )


    # example plot using plot_time_histories
    plt.figure("left-right muscles")
    plot_time_histories(
        controller.times,
        controller.state[:,left_idx]-controller.state[:,right_idx],
        offset=-0.,
        colors="green",
        savepath=log_path+"left-right_muscles.png",
        lw=1
        )


    # example plot using plot_time_histories
    plt.figure("link y-velocities_single")
    plot_time_histories(
        controller.times,
        controller.links_velocities[:,:,1],
        offset=-0.,
        colors="green",
        ylabel="link y-velocities",
        lw=1,
        savepath=log_path+"link_y-velocities.png",
        )


if __name__ == '__main__':
    main()



