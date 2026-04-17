"""Network"""

import numpy as np
from farms_ekeberg.src.network import NNController
from scipy import integrate

class PAOscillatorController(NNController):
    """Phase-amplitude Oscillator Controller"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations, config):

        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)

        self.n_links = self.animat_data.sensors.links.array.shape[1]
        self.freq    = 1


        # self.spine_left_idx    = range(0, self.n_joints)
        # self.spine_right_idx   = range(self.n_joints, 2*self.n_joints)
        # self.left_idx_a  = range(2*self.n_joints, 3*self.n_joints)
        # self.right_idx_a = range(3*self.n_joints, 4*self.n_joints)


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

        # self.phase_lag       = getattr(config, 'phase_lag', 2*np.pi/12)
        # self.weight          = getattr(config, 'weight', 5.0)
        # self.freq            = getattr(config, 'freq', 0.7)
        # self.enable_coupling = getattr(config, 'enable_coupling', 1)
        # self.weight_feedback = getattr(config, 'weight_feedback', 200.0)
        # self.amp_bias        = getattr(config, 'amp_bias', 0.0)
        # self.taua            = getattr(config, 'taua', 0.1)
        # self.go_straight     = getattr(config, 'go_straight', False)
        # initial_state        = getattr(config, 'initial_state', np.linspace(0, -2*np.pi, self.n_joints))

        # print("self.freq = ", self.freq)

        # shift_idx  = np.random.randint(0, n_joints)
        # init_state = np.roll(np.linspace(0, -2*np.pi, n_joints), shift_idx) + np.random.rand(n_joints) * 0.1

        # self.state[0][:self.n_joints]                = initial_state
        # self.state[0][self.n_joints:2*self.n_joints] = initial_state + np.pi


        # self.state[0][:2*self.n_joints] = np.random.rand(2*self.n_joints) * 2 * np.pi


        self.dstate   = np.zeros_like(self.state)
        self.solver   = integrate.ode(f=self.rhs)
        self.timestep = experiment_options.simulation.physics.timestep
        self.solver.set_integrator('dopri5', atol=1e-5, rtol=1e-5)
        self.solver.set_initial_value(y=self.state[0], t=0.0)

        self.animat_data.record = {
            "state"                 : self.state,
            "lateral_speed"         : np.zeros(self.n_iterations),
            "lateral_speed_filtered": np.zeros(self.n_iterations),
        }



    def couple(self, output, input, weight, phase_lag):
        """Unidirectional coupling function between oscillators"""
        return weight * np.sin(input - output - phase_lag)

    def rhs(self, time, state, iteration, joint_pos):
        """Right-hand side of the ODEs for the phase oscillators"""

        # self.dstate[iteration] = 2*np.pi*self.freq

        # phases_l = state[self.left_idx] # left phases
        # phases_r = state[self.right_idx] # right phases

        # # Phase dynamics
        # if self.enable_coupling:
        #     # down coupling
        #     self.dstate[iteration, self.left_idx[1:]]  += self.couple(phases_l[1:],phases_l[:-1], self.weight, self.phase_lag)
        #     self.dstate[iteration, self.right_idx[1:]] += self.couple(phases_r[1:],phases_r[:-1], self.weight, self.phase_lag)
        #     # up coupling
        #     self.dstate[iteration, self.left_idx[:-1]]  += self.couple(phases_l[:-1],phases_l[1:], self.weight, -self.phase_lag)
        #     self.dstate[iteration, self.right_idx[:-1]] += self.couple(phases_r[:-1],phases_r[1:], self.weight, -self.phase_lag)
        #     # contralateral coupling
        #     self.dstate[iteration, self.left_idx]  += self.couple(phases_l, phases_r, self.weight, np.pi)
        #     self.dstate[iteration, self.right_idx] += self.couple(phases_r, phases_l, self.weight, np.pi)
        # else:
        #     self.dstate[iteration, self.left_idx[0]]  += self.couple(phases_l[0], phases_r[0], self.weight, np.pi)
        #     self.dstate[iteration, self.right_idx[0]] += self.couple(phases_r[0], phases_l[0], self.weight, np.pi)

        # # Sensory feedback
        # joint_pos_spine = joint_pos[:self.n_joints]
        # # self.dstate[iteration, self.left_idx]  -= self.weight_feedback * np.maximum(joint_pos_spine,0) * np.sin(phases_l)
        # # self.dstate[iteration, self.right_idx] -= self.weight_feedback * np.maximum(-joint_pos_spine,0) * np.sin(phases_r)

        # # self.dstate[iteration, self.left_idx[1:]]  -= self.weight_feedback * np.maximum(joint_pos_spine[:-1],0) * np.sin(phases_l[1:])
        # # self.dstate[iteration, self.right_idx[1:]] -= self.weight_feedback * np.maximum(-joint_pos_spine[:-1],0) * np.sin(phases_r[1:])


        # joint_pos_spine = joint_pos[:self.n_joints]
        # self.dstate[iteration, self.left_idx[1:]]  -= self.weight_feedback * joint_pos_spine[:-1] * np.sin(phases_l[1:])
        # self.dstate[iteration, self.right_idx[1:]] += self.weight_feedback * joint_pos_spine[:-1] * np.sin(phases_r[1:])

        # # # Amplitude dynamics
        # # if self.amp_bias<0:
        # #     self.dstate[iteration, self.left_idx_a]  = (1+self.amp_bias-state[self.left_idx_a])/self.taua
        # #     self.dstate[iteration, self.right_idx_a] = (1-state[self.right_idx_a])/self.taua
        # # else:
        # #     self.dstate[iteration, self.left_idx_a]  = (1-state[self.left_idx_a])/self.taua
        # #     self.dstate[iteration, self.right_idx_a] = (1-self.amp_bias-state[self.right_idx_a])/self.taua

        # self.dstate[iteration, self.left_idx_a]  = (1+self.amp_bias-state[self.left_idx_a])/self.taua
        # self.dstate[iteration, self.right_idx_a] = (1-self.amp_bias-state[self.right_idx_a])/self.taua

        return self.dstate[iteration]

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
        if time>3:

            self.state[iteration, [8,20]]  = np.cos(self.get_y_from_phase(self.freq*(time - np.pi/4)))
            self.state[iteration, [12,16]] = np.cos(self.get_y_from_phase(self.freq*(time - np.pi/4 - np.pi)))
            self.state[iteration, [9,21]]  = np.maximum(np.cos(self.get_y_from_phase(self.freq*time)), -0)
            self.state[iteration, [13,17]] = np.maximum(np.cos(self.get_y_from_phase(self.freq*(time - np.pi))), -0)

        # self.state[iteration, self.idx_1] = np.sin(self.freq*time)
        # self.state[iteration, self.idx_2] = 0*np.sin(self.freq*time + np.pi/2)
        # self.state[iteration, self.idx_3] = 0*np.sin(self.freq*time + 3*np.pi/4)


        M_diff = self.state[iteration]
        M_sum  = np.zeros(self.n_joints)

        return M_diff, M_sum
