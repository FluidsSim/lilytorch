
"""Simulation parameters"""

import numpy as np

class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        """
        Class containing all neuromechanical model parameters
        Inputs:
        kwargs: extra parameter arguments (these override previous declarations)
        """
        super(SimulationParameters, self).__init__()

        # pars for the sine controller
        self.swimming_mode = "drag"
        self.amp           = 0.4*np.ones(15)
        self.bias          = 0.*np.ones(15)
        self.twl           = 0.8
        self.freq          = 1
        self.slope         = 100 # square scaling factor

        # simulation parameters
        self.n_joints     = 15 # number of joints
        self.timestep     = 0.001 # integration time step
        self.n_iterations = 4001 # number of integration time steps

        # gui/recording parameters
        self.headless        = False # For headless mode (No GUI, could be faster)
        self.fast            = False # For fast mode (not real-time)
        self.video_record    = False # For saving the video
        self.video_speed     = 1 # video speed
        self.log_path        = "" # path where the simulation data will be stored (no simulation data will be saved if the string is empty)
        self.video_name      = "test" # video name (saved in the log_path folder under the video_name name)
        self.video_type      = "mp4"
        self.video_fps       = 30 # frames per second
        self.camera_id       = 0 # camera type: 0 = angles top view, 1 = top view,       2 = side view,          3 = back view

        self.show_progress   = True # show progress bar of running the simulation
        self.simulation_i    = 0 # simulation id (of log_path!="" saves the simulation in the log_path folder under the name "simulation_i")
        self.compute_metrics = 0 # 0              = no metrics,      1 = neural metrics, 2 = mechanical metrics, 3 = all metrics (metrics are stored in network.metrics)
        self.print_metrics   = True # if True print all computed metrics
        self.return_network  = False # if True,    run_single_sim will return the controller class
                                     # (keep it False when running simulations on mutliple cpu cores,
                                     # or it will saturate the RAM memory)
        self.random_spine = False # if True, initialize the joints angles in a random position

        # parameters of the firing rate controller
        self.controller     = "sine" # "sine" for using the WaveController (Project 1), "firing_rate" for using the FiringRateController (Project 2)

        self.spawn = {
            'loader': 0,
            'mode': "TRANSVERSE",
            'pose': [0.0, 0.0, -0.03, 0.0, 0.0, 3.141592653589793],
            'velocity': [0.0, 0., 0.0, 0.0, 0.0, 0.0],
            'extras': {}
            }
        self.gravity = np.array([0,0,-9.81])

        self.__dict__.update(kwargs)  # NOTE: This overrides the previous declarations
