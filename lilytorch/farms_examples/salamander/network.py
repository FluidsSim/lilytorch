"""Network"""

import numpy as np
from farms_ekeberg.src.network import NNController
from scipy import integrate

class PAOscillatorController(NNController):
    """Phase-amplitude Oscillator Controller"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations, config):

        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)

        self.n_links   = self.animat_data.sensors.links.array.shape[1]
        self.freq      = 2*np.pi*0.7
        self.animat_id = config.animat_id

        self.idx_0=[]
        self.idx_1=[]
        self.idx_2=[]
        self.idx_3=[]
        for idx, motor in enumerate(experiment_options["animats"][0]["control"]["motors"]):
            if motor["joint_name"] in ("joint_leg_0_L_0", "joint_leg_0_R_0", "joint_leg_1_L_0", "joint_leg_1_R_0"):
                self.idx_0.append(idx)
            elif motor["joint_name"] in ("joint_leg_0_L_3", "joint_leg_0_R_3", "joint_leg_1_L_3", "joint_leg_1_R_3"):
                self.idx_1.append(idx)
            elif motor["joint_name"] in ("joint_leg_1_L_0", "joint_leg_1_R_0", "joint_leg_1_L_3", "joint_leg_1_R_3"):
                self.idx_2.append(idx)
            elif motor["joint_name"] in ("joint_leg_1_L_3", "joint_leg_1_R_3", "joint_leg_2_L_3", "joint_leg_2_R_3"):
                self.idx_3.append(idx)

        self.state  = np.zeros((self.n_iterations, self.n_joints))  # phases and amplitudes

        if self.animat_id == 0:
            self.init_delay = 0
        elif self.animat_id == 1:
            self.init_delay = np.pi/2
        elif self.animat_id == 2:
            self.init_delay = -np.pi/2


    def get_y_from_phase(self, phi, w=0.8):
        """
        Computes the analytical waveform y given the phase phi and duty cycle w.
        """
        sqrt_1_w2 = np.sqrt(1 - w**2)
        # Inner expression for arctan
        inner = w + sqrt_1_w2 * np.tan(0.5 * phi - np.arctan(w / sqrt_1_w2))
        return 2 * np.arctan(inner)

    def step(self, iteration, time, timestep):
        """Compute neural activity"""
        if time>0.:

            self.state[iteration, [8,20]]  = np.cos(self.get_y_from_phase(self.freq*time - np.pi/4 - self.init_delay, w=0.8))
            self.state[iteration, [12,16]] = np.cos(self.get_y_from_phase(self.freq*time - np.pi/4 - np.pi - self.init_delay, w=0.8))
            self.state[iteration, [9,21]]  = np.maximum(np.cos(self.get_y_from_phase(self.freq*time - self.init_delay, w=0.8)), 0)
            self.state[iteration, [13,17]] = np.maximum(np.cos(self.get_y_from_phase(self.freq*time - np.pi - self.init_delay, w=0.8)), 0)


        M_diff = self.state[iteration]
        M_sum  = np.zeros(self.n_joints)

        return M_diff, M_sum
