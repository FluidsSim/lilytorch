
from util.run_fluid_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import farms_pylog as pylog
import numpy as np

<<<<<<< HEAD
=======
REF_JOINT_AMP = 3*np.array([
    0.06580,
    0.02810,
    0.02781,
    0.03047,
    0.03623,
    0.04127,
    0.04864,
    0.05398,
    0.06508,
    0.08945,
    0.10271,
    0.11789,
    0.14929,
    0.0,      # Note: Tail moves passively,
    0.0,      # Note: Tail moves passively,
]) # type: ignore unit:radian

>>>>>>> cb21b7af07e09156fd8c754d1e4dcd698ed77036
def exercise_single(**kwargs):
    """
    Exercise example, running a single simulation and plotting the results
    """
<<<<<<< HEAD
    amps=np.ones(15)*1
=======
    amps=np.ones(15)*0.6
>>>>>>> cb21b7af07e09156fd8c754d1e4dcd698ed77036
    # amps[:5]=0
    bias=0.*np.ones(15)
    # bias[:5]=0

    all_pars = SimulationParameters(
        controller      = "sine",
        swimming_mode   = "bdim",
        amp             = amps,
        bias            = bias,
        wavefrequency   = 1,
        freq            = 5,
        n_iterations    = 80000,
<<<<<<< HEAD
        timestep        = 0.001,
=======
        timestep        = 0.0001,
>>>>>>> cb21b7af07e09156fd8c754d1e4dcd698ed77036
        compute_metrics = 3,
        n_joints        = 15,
        return_network  = True,
        headless        = True,
        **kwargs
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )



if __name__ == '__main__':
    exercise_single()
    plt.show()



