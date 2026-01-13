"""Network"""

import numpy as np
from farms_ekeberg.src.network import NNController
from scipy import integrate

class PAOscillatorController(NNController):
    """Phase-amplitude Oscillator Controller"""

    def __init__(self, animat_data, animat_options, experiment_options, n_joints, n_iterations, config):

        super().__init__(animat_data, animat_options, experiment_options, n_joints, n_iterations)

        self.n_links = self.animat_data.sensors.links.array.shape[1]

        self.left_idx    = range(0, self.n_joints)
        self.right_idx   = range(self.n_joints, 2*self.n_joints)
        self.left_idx_a  = range(2*self.n_joints, 3*self.n_joints)
        self.right_idx_a = range(3*self.n_joints, 4*self.n_joints)

        self.config = config
        self.state  = np.zeros((self.n_iterations, 4*self.n_joints))  # phases and amplitudes

        self.state[0][:2*self.n_joints] = np.random.rand(2*self.n_joints) * 2 * np.pi

        # self.state[0][:self.n_joints] = np.linspace(0, 2*np.pi, self.n_joints)
        # self.state[0][self.n_joints:2*self.n_joints] = 1 #np.linspace(0, 2*np.pi, self.n_joints) + np.pi

        self.phase_lag       = 2*np.pi/10
        self.weight          = 10.0
        self.freq            = 1
        self.enable_coupling = False

        self.go_straight     = True
        if self.go_straight:
            self.speed_lateral_filtered = 0.0 # initial value


        self.weight_feedback = 100.0
        self.amp_bias        = 0.0
        self.taua            = 0.1

        self.dstate          = np.zeros_like(self.state)
        self.solver          = integrate.ode(f=self.rhs)
        self.timestep        = experiment_options.simulation.physics.timestep
        self.solver.set_integrator('dopri5', atol=1e-10, rtol=1e-10)
        self.solver.set_initial_value(y=self.state[0], t=0.0)

        self.animat_data.record = {
            "state": self.state,
            "lateral_speed": np.zeros(self.n_iterations),
            "lateral_speed_filtered": np.zeros(self.n_iterations),
        }



    def couple(self, output, input, weight, phase_lag):
        """Unidirectional coupling function between oscillators"""
        return weight * np.sin(input - output - phase_lag)

    def rhs(self, time, state, iteration, joint_pos):
        """Right-hand side of the ODEs for the phase oscillators"""

        self.dstate[iteration] = 2*np.pi*self.freq

        phases_l = state[self.left_idx] # left phases
        phases_r = state[self.right_idx] # right phases

        if self.enable_coupling:
            # down coupling
            self.dstate[iteration, self.left_idx[1:]]  += self.couple(phases_l[1:],phases_l[:-1], self.weight, self.phase_lag)
            self.dstate[iteration, self.right_idx[1:]] += self.couple(phases_r[1:],phases_r[:-1], self.weight, self.phase_lag)

            # up coupling
            self.dstate[iteration, self.left_idx[:-1]]  += self.couple(phases_l[:-1],phases_l[1:], self.weight, -self.phase_lag)
            self.dstate[iteration, self.right_idx[:-1]] += self.couple(phases_r[:-1],phases_r[1:], self.weight, -self.phase_lag)
            # contralateral coupling
            self.dstate[iteration, self.left_idx]  += self.couple(phases_l, phases_r, self.weight, np.pi)
            self.dstate[iteration, self.right_idx] += self.couple(phases_r, phases_l, self.weight, np.pi)

        joint_pos_spine = joint_pos[:self.n_joints]

        self.dstate[iteration, self.left_idx[0]]  += self.couple(phases_l[0], phases_r[0], self.weight, np.pi)
        self.dstate[iteration, self.right_idx[0]] += self.couple(phases_r[0], phases_l[0], self.weight, np.pi)

        self.dstate[iteration, self.left_idx[1:]]  -= self.weight_feedback * joint_pos_spine[:-1] * np.sin(phases_l[1:])
        self.dstate[iteration, self.right_idx[1:]] += self.weight_feedback * joint_pos_spine[:-1] * np.sin(phases_r[1:])

        self.dstate[iteration, self.left_idx_a]  = (1+self.amp_bias-state[self.left_idx_a])/self.taua
        self.dstate[iteration, self.right_idx_a] = (1-self.amp_bias-state[self.right_idx_a])/self.taua

        return self.dstate[iteration]

    def step(self, iteration, time, timestep):
        """Compute neural activity"""

        joint_positions = np.array(self.animat_data.sensors.joints.positions(iteration))

        # set inputs to the ODEs - i.e. feedback from joints, forces, etc (for closed loop control)
        self.solver.set_f_params(iteration, joint_positions)
        self.state[iteration] = self.solver.integrate(time+timestep)
        if not self.solver.successful():
            message = (
                f'ODE not integrated properly at {iteration=}'
                f' ({self.solver.t=} < {time+timestep=} [s])'
                f'\nReturn code: {self.solver.get_return_code()=}'
            )
            print(message)

        if self.go_straight:

            # links_pos_y = np.array([self.animat_data.sensors.links.com_position(iteration, link_i)[1] for link_i in range(self.n_links)])
            # com_pos_y   = np.mean(links_pos_y)
            # var = com_pos_y

            link_vel_xy = np.array([self.animat_data.sensors.links.com_lin_velocity(iteration, link_i)[:2] for link_i in range(self.n_links)])
            v_com_xy    = np.mean(link_vel_xy,axis=0)
            v_com_y     = v_com_xy[1]

            # _, speed_lateral = self.compute_com_velocity(iteration)
            alpha                         = 2 * np.pi * self.freq * timestep
            self.speed_lateral_filtered  += alpha * (v_com_y - self.speed_lateral_filtered)
            self.amp_bias                 = -0.4 * self.speed_lateral_filtered
            print(v_com_y,self.amp_bias)

            self.animat_data.record["lateral_speed"][iteration] = v_com_y
            self.animat_data.record["lateral_speed_filtered"][iteration] = self.speed_lateral_filtered

        left_activities  = self.state[iteration, self.left_idx]
        right_activities = self.state[iteration, self.right_idx]

        M_left  = self.state[iteration, self.left_idx_a]*(1+np.cos(left_activities))
        M_right = self.state[iteration, self.right_idx_a]*(1+np.cos(right_activities))
        M_diff  = M_left-M_right
        M_sum   = M_right+M_left

        return M_diff, M_sum

    def compute_com_velocity(self, iteration):
        """Compute velocities of the COM using PCA method"""

        links_pos_xy = np.array([self.animat_data.sensors.links.com_position(iteration, link_i)[:2] for link_i in range(self.n_links)])
        link_vel_xy  = np.array([self.animat_data.sensors.links.com_lin_velocity(iteration, link_i)[:2] for link_i in range(self.n_links)])
        v_com_xy     = np.mean(link_vel_xy,axis=0)

        x = links_pos_xy[:,0]
        y = links_pos_xy[:,1]

        pheadtail = links_pos_xy[0]-links_pos_xy[-1]  # head - tail direction

        covmat               = np.cov([x,y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index        = np.argmax(eig_values)
        largest_eig_vec      = eig_vecs[:, largest_index]

        ht_direction       = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec    = ht_direction * largest_eig_vec

        speed_forward = np.dot(v_com_xy, largest_eig_vec)


        left_pointing_vec = np.cross(
            [0,0,1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        speed_lateral = np.dot(v_com_xy, left_pointing_vec)

        return speed_forward, speed_lateral

