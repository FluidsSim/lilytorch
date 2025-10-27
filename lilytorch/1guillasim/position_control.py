
from lilytorch.util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
from lilytorch.util.plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import farms_pylog as pylog
import numpy as np
from gen_positions import generate_positions

def main(amp,freq,twl,nmotors):
    """
    Exercise example, running a single simulation and plotting the results
    """
    log_path = './logs/drag_position_control/' # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    generate_positions(
        tstop=100,
        sampling_rate=1000,
        wlength=1,
        amp_deg=amp,
        freq=freq,
        nmotors=nmotors,
        TWL=twl,
        save_path=None,
        plot=False
    )

    all_pars = SimulationParameters(
        n_iterations    = 200001,
        controller      = "empty",
        swimming_mode   = "bdim",
        timestep        = 0.001,
        log_path        = log_path,
        compute_metrics = 0,
        headless        = True,
        return_network  = True,
        amp             = amp,
        freq            = freq,
        twl             = twl,
        gravity         = [0,0,-9.81],
        video_record    = False,
        video_name      = "swimming_example",
        yaml_path       = "models/1guilla_v1_position/",
        yaml_file       = "/data/andreaferrario/lilytorch/lilytorch/scripts/1guilla_swimming.yaml",
        spawn           = {
            'loader': 0,
            'mode': "TRANSVERSE",
            'pose': [0.0, 0.0, -0.3, 0.0, 0.0, 3.141592653589793],
            'velocity': [0.0, 0., 0.0, 0.0, 0.0, 0.0],
            'extras': {}
            },
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )

    # example plot using plot_time_histories_multiple_windows
    plt.figure("joint positions_single")
    plot_time_histories(
        controller.times,
        controller.joints_positions,
        offset=0.0,
        colors=plt.cm.jet( np.linspace( 0, 1, controller.joints_positions.shape[1])).tolist(),
        ylabel="joint positions",
        savepath=log_path+"joint_positions.png",
        lw=1
        )



if __name__ == '__main__':
    # main()
    freq=1
    twl=12 #14
    nmotors=8
    for amp in [30]: #np.arange(20,60,10):
        main(amp,freq,twl,nmotors)

