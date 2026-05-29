"""Network"""

import numpy as np
from farms_ekeberg.src.network import NNController
from lilytorch.farms_examples.zebrafish_ki_project.controller_parameters import (
    SLOW_SWIMMING_CONTROLLER_PARAMETERS,
    FAST_SWIMMING_CONTROLLER_PARAMETERS,
)

class WaveController(NNController):
    """Phase-amplitude Oscillator Controller"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations, config):

        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)

        self.n_links = self.animat_data.sensors.links.array.shape[1]

        if config.mode == "slow":
            self.freq = np.array(SLOW_SWIMMING_CONTROLLER_PARAMETERS["frequency"])
            self.ipl_arr = np.array(SLOW_SWIMMING_CONTROLLER_PARAMETERS["ipl_arr"])
            self.amp_arr = np.array(SLOW_SWIMMING_CONTROLLER_PARAMETERS["amp_arr"])
            self.bsl_arr = np.array(SLOW_SWIMMING_CONTROLLER_PARAMETERS["bsl_arr"])
        elif config.mode == "fast":
            self.freq = np.array(FAST_SWIMMING_CONTROLLER_PARAMETERS["frequency"])
            self.ipl_arr = np.array(FAST_SWIMMING_CONTROLLER_PARAMETERS["ipl_arr"])
            self.amp_arr = np.array(FAST_SWIMMING_CONTROLLER_PARAMETERS["amp_arr"])
            self.bsl_arr = np.array(FAST_SWIMMING_CONTROLLER_PARAMETERS["bsl_arr"])
        else:
            raise ValueError(f"Unknown mode {config.mode}. Expected 'slow' or 'fast'.")

        self.animat_data.record = {
            "M_diff"                 : np.zeros((self.n_iterations, self.n_joints)),
            "M_sum"                  : np.zeros((self.n_iterations, self.n_joints)),
        }


    def step(self, iteration, time, timestep):
        """Compute neural activity"""

        signals = np.sin( 2*np.pi * (self.freq * time - self.ipl_arr) )
        states_LR = np.array(
            [
                (
                    0.5 * j_bsl +                                        # Baseline  (M_L + M_R)
                    0.5 * j_sign * j_amp * signals[j_ind]          # Amplitude (M_L - M_R)
                )
                for j_ind, (j_amp, j_bsl) in enumerate(zip(self.amp_arr, self.bsl_arr))
                for j_sign in [+1, -1]
            ]
        )
        M_diff = states_LR[0::2] - states_LR[1::2]
        M_sum  = states_LR[0::2] + states_LR[1::2]

        return M_diff, M_sum
