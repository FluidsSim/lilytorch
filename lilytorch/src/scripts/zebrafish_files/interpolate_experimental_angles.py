
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
from lilytorch.scripts.zebrafish_files.data.experimental_angles import *

def interpolate_signal(times, signal, times_sampled):
    ''' Interpolates using cubic spline interpolation. '''
    func_interpolate    = CubicSpline(times, signal)
    # func_interpolate    = Akima1DInterpolator(times, signal)
    # func_interpolate    = interp1d(times, signal, kind='linear')
    interpolated_values = func_interpolate(times_sampled)
    return interpolated_values

# TODO: SIMILAR DATA EXTRACTION FOR THE DLC DATA OF ALL POINTS

def create_fictive_schooling_trace(
    timestep      : float,
    total_duration: float,
    freq_scaling  : float,
    time_offset   : float = 0.0,
    signal_repeats: int   = None,
    signal_only   : bool  = False,
    plot          : bool  = False,
):
    ''' Get scaled signal. '''

    # INTERPOLATE
    joint_angles_rad = np.deg2rad(JOINT_ANGLES)
    duration_scaling = 1 / freq_scaling
    sampling_times   = duration_scaling * SAMPLING_TIMES

    t_start = duration_scaling * ( SAMPLING_TIMES[0] )
    t_end   = duration_scaling * ( SAMPLING_TIMES[-1] )

    times_interpolated  = np.arange( t_start, t_end, timestep)
    angles_length       = len(times_interpolated)
    angles_interpolated = interpolate_signal(
        times         = sampling_times,
        signal        = joint_angles_rad,
        times_sampled = times_interpolated,
    )

    # SIGNAL ONLY
    if signal_only:
        signal_repeats     = int( np.ceil(total_duration / times_interpolated[-1]) )
        signal_onset_time  = 0.0 + time_offset
    else:
        signal_repeats = 1 if not signal_repeats else signal_repeats
        signal_duration   = times_interpolated[-1] * signal_repeats
        signal_onset_time = (total_duration - signal_duration) / 2 + time_offset

    signal_onset_index = round( signal_onset_time / timestep )

    # APPEND ZEROS
    times_angles = np.arange(0, total_duration + timestep, timestep)
    joint_angles = np.zeros_like(times_angles)

    for repeat in range(signal_repeats):
        i_start = signal_onset_index + repeat * angles_length
        i_end   = i_start + angles_length

        diff_start = max(0, -i_start)
        diff_end   = max(0, i_end - len(joint_angles))

        i_start += diff_start
        i_end   -= diff_end

        joint_angles[i_start: i_end] = angles_interpolated[diff_start: angles_length - diff_end]

    # PLOT
    if not plot:
        return joint_angles, times_angles

    plt.figure()
    plt.plot(
        times_angles,
        np.rad2deg(joint_angles),
        label     = 'Interpolated signal',
    )
    plt.plot(
        sampling_times,
        np.rad2deg(joint_angles_rad),
        label = 'Original signal',
        marker    = 'o',
        linestyle = '--',
        linewidth = 1.0,
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.show()


    return joint_angles, times_angles


if __name__ == '__main__':

    timestep       = 0.001
    total_duration = 30.0
    freq_scaling   = 0.30
    time_offset    = 0.0
    signal_repeats = None
    signal_only    = True
    plot           = True

    create_fictive_schooling_trace(
        timestep       = timestep,
        total_duration = total_duration,
        freq_scaling   = freq_scaling,
        time_offset    = time_offset,
        signal_repeats = signal_repeats,
        signal_only    = signal_only,
        plot           = plot,
    )
