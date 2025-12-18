
from lilytorch.util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog

def main():
    """
    Exercise example, running a single simulation and plotting the results
    """
    log_path = './logs/drag_position_control/' # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations    = 200001,
        controller      = "empty",
        swimming_mode   = "drag",
        timestep        = 0.001,
        log_path        = log_path,
        compute_metrics = 0,
        headless        = True,
        return_network  = True,
        gravity         = [0,0,-9.81],
        video_record    = False,
        video_name      = "",
        yaml_path       = "yamls/",
        # yaml_file       = "/data/andreaferrario/lilytorch/lilytorch/scripts/1guilla_swimming.yaml",
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



if __name__ == '__main__':
    main()

