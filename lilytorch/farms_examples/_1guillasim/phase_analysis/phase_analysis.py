
from lilytorch.util.paths import save_path

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, butter, filtfilt
import pywt
from scipy import ndimage

def get_schooling_data(
    times       : np.ndarray,
    joints_pos_1: np.ndarray,
    joints_pos_2: np.ndarray,
    links_pos_1 : np.ndarray,
    links_pos_2 : np.ndarray,
    linearize   : bool = False,
    plotting    : bool = True,
):
    ''' Save the COM position vs joint phase '''

    if times.shape[0] == joints_pos_1.shape[0] + 1:
        times = times[:-1]
    if times.shape[0] == joints_pos_2.shape[0] + 1:
        times = times[:-1]

    # Resample data to reduce computational cost
    resample_factor = 10
    times = times[::resample_factor]
    joints_pos_1 = joints_pos_1[::resample_factor]
    joints_pos_2 = joints_pos_2[::resample_factor]
    links_pos_1 = links_pos_1[::resample_factor]
    links_pos_2 = links_pos_2[::resample_factor]

    timestep = times[1] - times[0]

    # Get total angle evolution
    angle_1 = np.sum(joints_pos_1, axis=1)
    angle_2 = np.sum(joints_pos_2, axis=1)

    # Get com position evolution
    com_pos_1 = np.mean(links_pos_1, axis=1)
    com_pos_2 = np.mean(links_pos_2, axis=1)

    com_posx_1 = com_pos_1[:, 0]
    com_posx_2 = com_pos_2[:, 0]

    n         = angle_1.shape[0]
    normalize = True
    smoothing = False
    sigma     = 2

    freqs       = np.geomspace(1, 0.6, num=1000)
    wavelet     = 'cmor2.0-1.0'  # Good balance for frequencies 0.1-1 Hz
    scales      = 1 / (freqs*timestep)
    frequencies = pywt.scale2frequency(wavelet, scales) / timestep
    scaleMatrix = np.ones([1, n]) * scales[:, None]

    # step 1: compute the cwt and maxial power frequency of the normalized signals
    def compute_cwt(signal):
        if normalize:
            signal = (signal - signal.mean()) / signal.std()
        [W, _] = pywt.cwt(
            signal,
            scales,
            wavelet,
            timestep,
            method="fft"
            )
        S = np.abs(W**2)/scaleMatrix
        if smoothing:
            S = ndimage.gaussian_filter(S, sigma=sigma)

        freq_idx = np.argmax(S,axis=0)
        power_1d = S[freq_idx,range(n)]
        # freq_max = np.where(power_1d>0.1, frequencies[freq_idx], 0)
        freq_max = frequencies[freq_idx]
        return W, freq_max, power_1d

    # step 2: compute cross-wavelet transform and the maximal power phase difference angle
    def compute_cross_coherence(W1, W2, smoothing=smoothing):
        xwt       = W1 * W2.conj()
        power_xwt = np.abs(xwt**2)/scaleMatrix
        if smoothing:
            power_xwt = ndimage.gaussian_filter(power_xwt, sigma=sigma)
            # power_xwt = ndimage.gaussian_filter1d(power_xwt, sigma, 1)
        phase_xwt = np.angle(xwt)
        freq_idx  = np.argmax(power_xwt,axis=0)
        phi_max   = phase_xwt[freq_idx,range(n) ]
        return phi_max



    W_1, freq_1, power_1 = compute_cwt(angle_1)
    W_2, freq_2, power_2 = compute_cwt(angle_2)


    phases = compute_cross_coherence(W_1, W_2)


    if plotting:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        # Plot freq_1 and freq_2
        axes[0].plot(times, freq_1, label='Agent 1', color='blue', linewidth=1.5)
        axes[0].plot(times, freq_2, label='Agent 2', color='red', linewidth=1.5)
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title('Dominant Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot angle_1 and angle_2
        axes[1].plot(times, angle_1, label='Agent 1', color='blue', linewidth=1.5)
        axes[1].plot(times, angle_2, label='Agent 2', color='red', linewidth=1.5)
        axes[1].set_ylabel('Angle (rad)')
        axes[1].set_title('Joint Angles')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot limb phases
        axes[2].plot(times, phases, label='Phase Difference', color='green', linewidth=1.5)
        axes[2].set_ylabel('Phase (rad)')
        axes[2].set_title('Limb Phase Difference')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Hide unused subplot
        axes[3].axis('off')


        plt.tight_layout()



        plt.show()
    return times, freq_1, freq_2, phases





    # Convert joint angles to phases
    hilb_angle_1 = hilbert(angle_1)
    hilb_angle_2 = hilbert(angle_2)

    phase_1 = np.unwrap(np.angle(hilb_angle_1))
    phase_2 = np.unwrap(np.angle(hilb_angle_2))

    # Linearize phases
    linearize = True

    if linearize:
        phase_1 = _linearize_phases(times, phase_1)
        phase_2 = _linearize_phases(times, phase_2)

    # Align phases to second peak
    peaks_1 = find_peaks( angle_1 )[0]
    peaks_2 = find_peaks( angle_2 )[0]

    n_peaks_1 = len(peaks_1)
    n_peaks_2 = len(peaks_2)
    first_peak_ind = 1

    if n_peaks_1 > first_peak_ind:
        first_peak_1    = peaks_1[first_peak_ind]
        phase_1 -= phase_1[first_peak_1]

    if n_peaks_2 > first_peak_ind:
        first_peak_2    = peaks_2[first_peak_ind]
        phase_2 -= phase_2[first_peak_2]

    # Convert COM positions to body length units
    com_posx_1 = com_posx_1 / body_length
    com_posx_2 = com_posx_2 / body_length

    # Data to compare
    com_posx_diff = com_posx_1 - com_posx_2
    phase_diff    = phase_1 - phase_2

    # Normalize phases
    phase_diff = wrap_phase_to_pi_pi(phase_diff)

    # Integral of the phase difference
    phi1                = phase_1
    phi2                = phase_2
    n_cycles            = (phi2[-1] - phi2[0]) / (2*np.pi)
    n_cycles_ref        = (phi1[-1] - phi1[0]) / (2*np.pi)
    freq_lockking_ratio = n_cycles / n_cycles_ref

    logging.info(f"Frequency locking ratio: {freq_lockking_ratio * 100:.2f}%")

    ########################################################
    # All Data #############################################
    ########################################################

    schooling_data = pd.DataFrame(
        {
            'times'        : times[:-1],
            'angle_1'      : angle_1,
            'angle_2'      : angle_2,
            'phase_1'      : phase_1,
            'phase_2'      : phase_2,
            'com_posx_1'   : com_posx_1,
            'com_posx_2'   : com_posx_2,
            'com_posx_diff': com_posx_diff,
            'phase_diff'   : phase_diff,

        }
    )

    return schooling_data

###############################################################################
# PLOTTING ####################################################################
###############################################################################

def _plot_angle_vs_angle(
    times       : np.ndarray,
    angle_1     : np.ndarray,
    angle_2     : np.ndarray,
    figures_dict: dict[str, plt.Figure],
    n_steps     : int = None,
):
    ''' Plot joint angles evolution '''


    # Considered steps
    n_steps = n_steps if n_steps else len(times)
    times   = times[-n_steps:]
    angle_1 = angle_1[-n_steps:]
    angle_2 = angle_2[-n_steps:]

    ## Angles-time plot
    fig_1 = plt.figure('Joint angle vs Reference angle - 1D')
    axis  = fig_1.add_subplot(111)

    # Plot data
    axis.plot(times, angle_1, label='Reference')
    axis.plot(times, angle_2, label='Follower')

    # Decorate
    min_angle = np.amin([angle_1, angle_2])
    max_angle = np.amax([angle_1, angle_2])

    axis.set_xlim([times[0], times[-1]])
    axis.set_ylim([min_angle, max_angle])
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Joint Angle [rad]')
    axis.set_title('Joint Angle vs Reference Angle')
    axis.legend()

    figures_dict['fig_joint_angle_vs_ref_angle_1D'] = fig_1

    ## Angle-Angle plot
    fig_2 = plt.figure('Joint angle vs Reference angle - 2D')
    axis  = fig_2.add_subplot(111)

    # Plot data
    axis.plot( angle_1, angle_2 )

    # Plot diagonal line
    min_angle = np.amin( [ angle_1, angle_2, ] )
    max_angle = np.amax( [ angle_1, angle_2, ] )
    axis.plot([min_angle, max_angle], [min_angle, max_angle], 'k--')

    # Decorate
    axis.set_xlim([min_angle, max_angle])
    axis.set_ylim([min_angle, max_angle])
    axis.set_xlabel('Reference Angle [rad]')
    axis.set_ylabel('Joint Angle [rad]')
    axis.set_title('Joint Angle vs Reference Angle')

    figures_dict['fig_joint_angle_vs_ref_angle_2D'] = fig_2

    return

def _plot_phase_vs_phase(
    times       : np.ndarray,
    phase_1     : np.ndarray,
    phase_2     : np.ndarray,
    figures_dict: dict[str, plt.Figure],
    n_steps     : int = None,
):
    ''' Plot joint angles evolution '''

    # Considered steps
    n_steps = n_steps if n_steps else int(len(times)/2)
    times   = times[-n_steps:]
    phase_1 = phase_1[-n_steps:]
    phase_2 = phase_2[-n_steps:]

    ### PHASES
    fig_1 = plt.figure('Joint Phase vs Reference Phase')
    axis  = fig_1.add_subplot(111)

    phase_1_norm = wrap_phase_to_0_2pi(phase_1)
    phase_2_norm = wrap_phase_to_0_2pi(phase_2)

    axis.plot( times, phase_1_norm, label = 'Reference Phase' )
    axis.plot( times, phase_2_norm, label = 'Joint Phase' )

    # Decorate
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Phase [rad]')
    axis.set_title('Joint Phase vs Reference Phase')
    axis.legend()

    figures_dict['fig_joint_phase_vs_ref_phase'] = fig_1

    ### PHASE DIFFERENCE
    fig_2      = plt.figure('Joint Phase vs Reference Phase - Difference')
    phase_diff = wrap_phase_to_05_05( phase_1 - phase_2 )

    plt.plot(times, phase_diff, 'k-')
    plt.xlim(times[0], times[-1])
    plt.ylim(np.amin(phase_diff), np.amax(phase_diff))
    plt.xlabel('Time [s]')
    plt.ylabel('Phase Difference [cycles]')
    plt.title('Phase Difference Over Time')

    figures_dict['fig_joint_phase_vs_ref_phase_diff'] = fig_2

    return

def _plot_histogram(
    x_values  : np.ndarray,
    phi_values: np.ndarray,
    fig       : plt.Figure,
    ax        : plt.Axes,
):
    ''' Plot histogram of joint angles evolution '''

    # X bins
    x0      = np.amin(x_values)
    x1      = np.amax(x_values)
    x_range = x1 - x0
    x_min   = min( 0.4, x0 - 0.1*x_range )
    x_max   = max( 1.6, x1 + 0.1*x_range )
    x_bins  = 41
    x_step  = ( x_max - x_min ) / ( x_bins - 1 )
    x_range = ( x_min - x_step / 2, x_max + x_step / 2 )

    # Phi bins
    phi_min   = -np.pi
    phi_max   = 5 * np.pi
    phi_bins  = 40
    phi_step  = (phi_max - phi_min) / ( phi_bins - 1 )
    phi_range = ( phi_min - phi_step / 2, phi_max + phi_step / 2)

    # Histogram
    histogram, x_edges, phi_edges = np.histogram2d(
        x     = x_values,
        y     = phi_values,
        bins  = [ x_bins,  phi_bins],
        range = [x_range, phi_range],
    )

    # Normalize the histogram bt the total number of iterations
    n_iterations         = x_values.shape[0] / 3
    max_density          = 0.25 * n_iterations
    normalized_histogram = histogram / max_density

    # Account for periodicity
    hist_aux = normalized_histogram[:, 0] + normalized_histogram[:, -1]
    normalized_histogram[:,  0] = hist_aux
    normalized_histogram[:, -1] = hist_aux

    # Plot the histogram
    mesh_x, mesh_Y = np.meshgrid(x_edges, phi_edges)
    pcm = ax.pcolormesh(
        mesh_x,
        mesh_Y,
        normalized_histogram.T,
        cmap    = 'hot',
        shading = 'auto',
        vmin    = 0.0,
        vmax    = 1.0,
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Phase matching ratio", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add dashed lines at multiples of pi
    for i in range(-1, 6):
        plt.axhline(i * np.pi, color='white', linestyle='--', linewidth=0.8)

    # Put ylabels and ticks at multiples of pi
    plt.yticks(
        ticks  = np.arange(-np.pi, 5*np.pi+1, np.pi),
        labels = [ r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$', r'$5\pi$']
    )

    # Increas thickness of ticks
    plt.tick_params(axis='both', which='major', width=1.5)

    # Label axes
    plt.xlabel("FB distance (bl)", fontsize=14)
    plt.ylabel("Phase difference $\\Delta \\Phi$", fontsize=14)
    plt.title("$\\Delta \\Phi = \\Phi_L - \\Phi_F$", fontsize=16)

    # Set limits
    plt.xlim(x_range)
    plt.ylim(phi_range)

    return fig, ax

def _plot_position_vs_phase(
    com_posx_diff: np.ndarray,
    phase_diff   : np.ndarray,
    figures_dict : dict[str, plt.Figure],
    n_steps      : int = None,
):
    ''' Plot joint angles evolution '''

    # Considered steps
    n_steps       = n_steps if n_steps else len(phase_diff)
    com_posx_diff = com_posx_diff[-n_steps:]
    phase_diff    = phase_diff[-n_steps:]

    ###############
    ## 1D Histogram
    ###############
    fig_1 = plt.figure('COM Position vs Joint Phase Difference - 1D')

    plt.hist(
        wrap_phase_to_05_05(phase_diff),
        bins      = 80,
        color     = 'royalblue',
        edgecolor = 'black',
        alpha     = 0.75,
        linewidth = 1.2,
    )

    plt.xlabel('Joint Phase Difference [% cycle]', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of Joint Phase Difference', fontsize=14)
    plt.xlim([-0.5, 0.5])
    plt.tight_layout()

    figures_dict['fig_com_vs_phase_1D'] = fig_1

    ###############
    ## 2D Histogram
    ###############
    fig_2 = plt.figure('COM Position vs Joint Phase Difference - 2D')
    axis  = fig_2.add_subplot(111)

    # Stack copies for histogram
    com_positions_stack = _get_stacked_com_pos_for_histogram(com_posx_diff)
    joints_phases_stack = _get_stacked_phases_for_histogram(phase_diff)

    fig_2, axis = _plot_histogram(
        x_values  = com_positions_stack,
        phi_values= joints_phases_stack,
        fig       = fig_2,
        ax        = axis,
    )

    figures_dict['fig_com_vs_phase_2D'] = fig_2

    return

def plot_schooling_data(
    schooling_data : pd.DataFrame,
    show_plots     : bool = True,
):
    ''' Plot joint angles evolution '''

    # Parameters
    times        = schooling_data['times'].to_numpy()
    figures_dict = {}

    # Data
    angle_1 = schooling_data['angle_1'].to_numpy()
    angle_2 = schooling_data['angle_2'].to_numpy()
    phase_1 = schooling_data['phase_1'].to_numpy()
    phase_2 = schooling_data['phase_2'].to_numpy()

    com_posx_1    = schooling_data['com_posx_1'].to_numpy()
    com_posx_2    = schooling_data['com_posx_2'].to_numpy()
    com_posx_diff = schooling_data['com_posx_diff'].to_numpy()
    phase_diff    = schooling_data['phase_diff'].to_numpy()

    # Joint Angle vs Reference Angle #######################
    _plot_angle_vs_angle(
        times        = times,
        angle_1      = angle_1,
        angle_2      = angle_2,
        figures_dict = figures_dict,
    )

    # Joint Phase vs Reference Phase #######################
    _plot_phase_vs_phase(
        times        = times,
        phase_1      = phase_1,
        phase_2      = phase_2,
        figures_dict = figures_dict,
    )

    # COM vs Joint Phase ###################################
    _plot_position_vs_phase(
        com_posx_diff = com_posx_diff,
        phase_diff    = phase_diff,
        figures_dict  = figures_dict,
    )

    if show_plots:
        _gentle_plt_show(user_input=True)

    return
