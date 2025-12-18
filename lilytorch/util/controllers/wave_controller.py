"""Network controller"""

import numpy as np

class WaveController:

    """Test controller"""
    def __init__(self, pars):
        self.pars     = pars
        self.times    = np.linspace(0, pars.n_iterations*pars.timestep, pars.n_iterations)
        self.n_total_joints = 8
        self.state    = np.zeros((pars.n_iterations, 2*self.n_total_joints)) # state array for recording all the variables
        self.remove_first  = 0
        self.muscle_l = 2*np.arange(0, self.n_total_joints) # indexes of the left muscle activations (optional)
        self.muscle_r = self.muscle_l+1 # indexes of the right muscle activations (optional)
        self.pars.amplitudes_left = pars.amp+pars.bias
        self.pars.amplitudes_right = pars.amp-pars.bias

    def step(self, iteration, time, timestep, pos=None, urdf_positions=None):

        time     = iteration * timestep
        aux_sine = np.sin(
            2*np.pi * ( self.pars.freq*time - self.pars.twl*np.arange(self.n_total_joints)/self.n_total_joints )
        )

        # New motor output
        self.state[iteration, self.muscle_l]  = self.pars.amplitudes_left * aux_sine/2
        self.state[iteration, self.muscle_r]  = -self.pars.amplitudes_left * aux_sine/2

        return self.state[iteration,:]


