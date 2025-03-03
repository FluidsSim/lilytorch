"""Network controller"""

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.stats as ss
import farms_pylog as pylog

class FiringRateController:
    """zebrafish controller"""

    def __init__(
            self, 
            pars
            ):
        super().__init__()

        self.n_joints       = pars.n_joints
        self.n_iterations   = pars.n_iterations
        self.n_neurons      = pars.n_neurons
        self.n_muscle_cells = pars.n_muscle_cells
        self.timestep       = pars.timestep
        self.times          = np.linspace(0, self.n_iterations*self.timestep, self.n_iterations)
        self.pars           = pars

        self.n_eq        = self.n_neurons*4 + self.n_muscle_cells*2 + self.n_neurons*2 # number of equations: number of CPG eq+muscle cells eq+sensors eq
        self.muscle_l    = 4*self.n_neurons + 2*np.arange(0, self.n_muscle_cells) # muscle cells left indexes
        self.muscle_r    = self.muscle_l+1 # muscle cells right indexes
        self.all_muscles = 4*self.n_neurons + np.arange(0, 2*self.n_muscle_cells) # all muscle cells indexes
        # self.all_v       = range(self.n_neurons*2) # vector of indexes for the CPG activity variables - modify this according to your implementation # _S # _u

        # pylog.warning("Implement here the vectorization indexed for the equation variables") # _S # _u

        # _C0
        # build index vectors used for vectorized computations in numpy
        self.left_v      = 4*np.arange(0, self.n_neurons)
        self.left_a      = self.left_v+1
        self.right_v     = self.left_v+2
        self.right_a     = self.left_v+3
        self.all_v       = np.concatenate([self.left_v,self.right_v])
        self.all_a       = np.concatenate([self.left_a,self.right_a])
        self.sensors_l   = 4*self.n_neurons + 2*self.n_muscle_cells + 2*np.arange(0, self.n_neurons)
        self.sensors_r   = self.sensors_l+1
        self.all_sensors = np.concatenate([self.sensors_l,self.sensors_r])
        # _C1

        self.state    = np.zeros([self.n_iterations, self.n_eq]) # equation state
        self.dstate   = np.zeros([self.n_eq]) # derivative state
        self.state[0] = 0.01*np.random.rand(self.n_eq) # set random initial state

        # _C0
        # initialize connectivity matrices
        self.A_V0d2V0d    = self.pars.w_inh*self.A_generic(self.n_neurons, self.pars.n_asc, self.pars.n_desc).T
        self.A_V2a2muscle = self.pars.w_V2a2muscle*self.V2a2muscle(self.n_neurons, self.n_muscle_cells).T
        self.A_src2cpg    = self.A_generic(self.n_neurons, self.pars.n_asc_str, self.pars.n_desc_str).T

        # initialize input vector
        self.inputs = np.zeros(self.n_eq)
        # _C1

        self.poses = np.array([
            0.007000000216066837,
            0.00800000037997961,
            0.008999999612569809,
            0.009999999776482582,
            0.010999999940395355,
            0.012000000104308128,
            0.013000000268220901,
            0.014000000432133675,
            0.014999999664723873,
            0.01600000075995922,
        ]) # active joint distances along the body (pos=0 is the tip of the head)
        self.poses_ext = np.linspace(self.poses[0],self.poses[-1],self.n_neurons) # position of the sensors

        # initialize ode solver
        self.f = self.ode_rhs

        # stepper function selection
        if self.pars.method=="euler":
            self.step = self.step_euler
        elif self.pars.method=="noise":
            self.step = self.step_euler_maruyama
            self.noise_vec = np.zeros(self.n_neurons*2) # vector of noise for the CPG voltage equations (2*n_neurons)

        # zero vector activations to make first and last joints passive 
        self.zeros8 = np.zeros(8) # pre-computed zero activity for the first 4 joints
        self.zeros2 = np.zeros(2) # pre-computed zero activity for the tail joint

    def get_ou_noise_process_dw(self, timestep, x_prev, sigma):
        """
        Implement here the integration of the Ornstein-Uhlenbeck processes 
        dx_t = -0.5*x_t*dt+sigma*dW_t
        Parameters
        ----------
        timestep: <float>
            Timestep
        x_prev: <np.array>
            Previous time step OU process
        sigma: <float>
            noise level
        Returns
        -------
        x_t{n+1}: <np.array>
            The solution x_t{n+1} of the Euler Maruyama scheme
            x_new = x_prev-0.1*x_prev*dt+sigma*sqrt(dt)*Wiener
        """ 

        # _C0
        n_processes = len(x_prev)
        w_distrib   = ss.norm.rvs(loc=0, scale=sigma*np.sqrt(timestep), size=(n_processes))
        dx_process  = -0.1*x_prev*timestep + w_distrib
        return dx_process
        # _C1
        # dx_process  = np.zeros_like(x_prev) # _S # _u

    def step_euler(self, iteration, time, timestep, pos=None, urdf_positions=None):
        """Euler step"""
        self.state[iteration+1, :] = self.state[iteration, :] + timestep*self.f(time, self.state[iteration], pos=pos, urdf_positions=urdf_positions)
        return np.concatenate([
            self.zeros8, # the first 4 passive joints
            self.motor_output(iteration), # the active joints
            self.zeros2 # the last (tail) passive joint
            ])
        

    def step_euler_maruyama(self, iteration, time, timestep, pos=None, urdf_positions=None):
        """Euler Maruyama step"""
        self.state[iteration+1, :] = self.state[iteration, :] + timestep*self.f(time, self.state[iteration], pos=pos, urdf_positions=None)
        self.noise_vec = self.get_ou_noise_process_dw(timestep, self.noise_vec, self.pars.noise_sigma)
        self.state[iteration+1, self.all_v] += self.noise_vec
        self.state[iteration+1, self.all_muscles] = np.maximum(self.state[iteration+1, self.all_muscles], 0) # prevent from negative muscle activations
        return np.concatenate([
            self.zeros8, # the first 4 passive joints
            self.motor_output(iteration), # the active joints
            self.zeros2 # the last (tail) passive joint
            ])
    
    def motor_output(self, iteration):
        """
        Here you have to final muscle activations for the 10 active joints. 
        It should return an array of 2*n_muscle_cells=20 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations
        """
        return self.pars.act_strength*self.state[iteration+1,self.all_muscles] # _C
        # return np.zeros(2*self.n_muscle_cells) # here you have to final active muscle equations for the 10 joints  # _S # _u
    
    def ode_rhs(self,  _time, state, pos=None, urdf_positions=None):
        """Network_ODE
        You should implement here the right hand side of the system of equations
        Parameters
        ----------
        _time: <float>
            Time
        state: <np.array>
            ODE states at time _time
        Returns
        -------
        dstate: <np.array>
            Returns derivative of state
        """ 
        self.inputs[self.all_v] = self.pars.I-self.pars.b*state[self.all_a]

        self.inputs[self.left_v]  += + self.pars.Idiff \
                                    - self.A_V0d2V0d @ state[self.right_v]  \
                                    - self.pars.w_stretch*self.A_src2cpg @ state[self.sensors_r] 
        self.inputs[self.right_v] += - self.pars.Idiff \
                                    - self.A_V0d2V0d @ state[self.left_v] \
                                    - self.pars.w_stretch*self.A_src2cpg @ state[self.sensors_l] 

        # voltage neural equations
        self.dstate[self.all_v] = ( -state[self.all_v] + self.S( self.inputs[self.all_v] ) ) / self.pars.tau

        # adaptation neural equations
        self.dstate[self.all_a] = ( -state[self.all_a] + self.pars.gamma * state[self.all_v] ) / self.pars.taua

        # muscle cells equations
        self.dstate[self.muscle_l] = self.A_V2a2muscle @ state[self.left_v] * (1-state[self.muscle_l]) / self.pars.taum_a \
                                    - state[self.muscle_l] / self.pars.taum_d
        self.dstate[self.muscle_r] = self.A_V2a2muscle @ state[self.right_v] *  (1-state[self.muscle_r]) / self.pars.taum_a \
                                    - state[self.muscle_r] / self.pars.taum_d

        if pos is not None:
            func_interpolate = CubicSpline(self.poses, pos)
            poses_ext = func_interpolate(self.poses_ext)
            # stretch sensors - theta>0 is left bending(stretch), theta<0 is right bending(stretch)
            self.dstate[self.sensors_l] = ( -state[self.sensors_l] + (1-state[self.sensors_l])*self.S( poses_ext ) ) / self.pars.tau_str
            self.dstate[self.sensors_r] = ( -state[self.sensors_r] + (1-state[self.sensors_r])*self.S( -poses_ext ) ) / self.pars.tau_str
        return self.dstate
        # return self.dstate # _S # _u

    # _C0
    def A_generic(self, n_neurons, asc_n, desc_n):
        """
        A[i,j] = connection from i to j
        """
        A=np.zeros((n_neurons,n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i<=j and j-i<=desc_n: # descending
                    A[i,j]=1/(j-i+1)
                elif i>j and i-j<=asc_n: # ascending
                    A[i,j]=1/(i-j+1)
        return A
    
    def V2a2muscle(self, n_neurons, n_muscles):
        """
        A[i,j] = connection from i to j
        """
        A_V2a2muscle=np.zeros((n_neurons,n_muscles))
        if n_muscles>0:
            n_size = n_neurons/n_muscles
            if not n_size.is_integer():
                raise Exception("n_neurons/n_muscles is not an integer")
            for j in range(n_muscles):
                for i in range(int(n_size)*j,int(n_size)*(j+1)):
                    A_V2a2muscle[i, j]=1 
        return A_V2a2muscle

    def S(self,x):
        """
        gain function
        """
        return np.sqrt(np.maximum(x,0.))
    # _C1

