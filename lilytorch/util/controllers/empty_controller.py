"""Network controller"""

import numpy as np

class EmptyController:

    """Test controller"""
    def __init__(self, pars):
        self.pars     = pars
        self.timestep = pars.timestep
        self.times    = np.linspace(0, pars.n_iterations*pars.timestep, pars.n_iterations)
        self.state    = np.zeros((pars.n_iterations, 16)) # state array for recording all the variables

    def step(self, iteration, time, timestep, pos=None, urdf_positions=None):
        """
        Step function. This function passes the activation functions of the muscle model
        Inputs:
        - iteration - iteration index
        - time - time vector
        - timestep - integration timestep
        - pos (not used) - joint angle positions

        Implement here the control step function,
        it should return an array of 2*n_joint=30 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations

        In addition to returning the activation functions, store
        them in self.state for later use offline
        """


        return self.state[iteration,:]
