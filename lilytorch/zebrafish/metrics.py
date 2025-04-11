

"""
File of metrics computations
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, fftconvolve
import sys


POINTS_POS = np.array(
    [
        0.000,
        0.003, 0.004, 0.005,
        0.006, 0.007, 0.008,
        0.009, 0.010, 0.011,
        0.012, 0.013, 0.014,
        0.015, 0.016, 0.017,
        0.018,
    ]
)

BODY_LENGTH   = POINTS_POS[-1] - POINTS_POS[0]
JOINTS_POS    = POINTS_POS[1:-1]
LINKS_COM_POS = ( POINTS_POS[1:] + POINTS_POS[:-1] ) / 2

LINKS_MASSES = np.array(
    [
        3.27e-06, 4.05e-06, 4.85e-06, 5.15e-06,
        5.49e-06, 5.47e-06, 5.15e-06, 4.24e-06,
        3.40e-06, 2.65e-06, 1.73e-06, 1.17e-06,
        8.25e-07, 9.19e-07, 9.36e-07, 8.46e-07,
    ]
)

def travel_distance(links_data, sim_fraction=1.0):
    """Compute total travel distance, regardless of its curvature"""
    nsteps = links_data.shape[0]
    nsteps_considered = round(nsteps * sim_fraction)

    com = np.mean( links_data[-nsteps_considered:], axis= 1)

    comx = com[:,0]
    comy = com[:,1]

    return np.sum( np.sqrt( np.diff( comx )**2 + np.diff( comy )**2 ) )


def compute_curvature(
    network,
    sim_fraction = 1.0
):
    com_pos = np.mean(network.links_positions[:-1, :, :2], axis=1)
    nsteps = com_pos.shape[0]
    nsteps_considered = round(nsteps * sim_fraction)
    com_pos = com_pos[-nsteps_considered:, :]


    # compute freq based on the state
    times = network.times[-nsteps_considered:]
    state = network.state[-nsteps_considered:]
    muscle_diff = state[:,network.muscle_l]-state[:,network.muscle_r]

    frequency, _ = compute_frequency_treshold_crossings(times, muscle_diff.T, theta = 0.0)

    if frequency>0:

        sampling_rate = int(1/(frequency*network.pars.timestep))
        sampling_idxs = np.arange(0,nsteps_considered,sampling_rate)
        nsamples = len(sampling_idxs)

        curvatures_sampled    = np.zeros(nsamples-1)
        for idx in range(nsamples-2):

            sampled_pos   = com_pos[sampling_idxs[idx]:sampling_idxs[idx+1],:]

            nsteps  = sampled_pos.shape[0]
            dx_dt   = np.gradient(sampled_pos[:, 0])
            dy_dt   = np.gradient(sampled_pos[:, 1])
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
            curvatures_sampled[idx+1] = np.mean(curvature)

        return curvatures_sampled
    else:
        return 0.0


def sum_torques(joints_data, sim_fraction=1.0):
    """Compute sum of torques"""
    nsteps = joints_data.shape[0]
    nsteps_considered = round(nsteps * sim_fraction)
    return np.sum(np.abs(joints_data[-nsteps_considered:, :]))


def compute_speed_cycle(
    network,
    sim_fraction : float = 1.0,
    ):
    """
    Compute forward and lateral speeds using the xy joint positions
    sampled across cycles using the frequency computed using the
    treshold crossing method
    """

    n_steps            = network.links_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)
    links_pos_xy       = network.links_positions[-n_steps_considered:,:,:2]

    # compute freq based on the state
    times = network.times[-n_steps_considered:]
    state = network.state[-n_steps_considered:]
    muscle_diff = state[:,network.muscle_l]-state[:,network.muscle_r]

    frequency, _ = compute_frequency_treshold_crossings(times, muscle_diff.T, theta = 0.0)

    if frequency>0:


        sampling_rate = int(1/(frequency*network.pars.timestep))
        sampling_idxs = np.arange(0,n_steps_considered,sampling_rate)
        sampled_pos   = links_pos_xy[sampling_idxs,0,:] # head pos

        vel = frequency*(sampled_pos[1:]-sampled_pos[:-1]) # head velocity

        nsamples = len(sampling_idxs)
        v_fwd    = np.zeros(nsamples-1)
        v_lat    = np.zeros(nsamples-1)
        for idx in range(nsamples-2):
            #
            p_head_fwd0 = links_pos_xy[sampling_idxs[idx],0,:]-links_pos_xy[sampling_idxs[idx],1,:]
            move_direction0 = p_head_fwd0/np.linalg.norm(p_head_fwd0)
            angles = []
            theta = np.arctan2(move_direction0[1], move_direction0[0])

            for idx2 in range(sampling_idxs[idx]+1, sampling_idxs[idx+1]):
                # to avoid the bias from the arctan switching sign at theta=pi we sum across small dthetas in each iteration
                p_head_fwd = links_pos_xy[idx2,0,:]-links_pos_xy[idx2,1,:]
                move_direction = p_head_fwd/np.linalg.norm(p_head_fwd)

                cos_dtheta = np.dot(move_direction0, move_direction)
                sin_dtheta = np.cross(move_direction0, move_direction)
                dtheta = np.arctan2(sin_dtheta, cos_dtheta)

                theta = theta + dtheta
                angles.append(theta)
                move_direction0 = move_direction


            mean_angle   = np.mean(angles)
            ht_direction = np.array([np.cos(mean_angle), np.sin(mean_angle)])
            ht_direction = ht_direction/np.linalg.norm(ht_direction)
            vec_left           = np.cross(
                [0,0,1],
                [-ht_direction[0], -ht_direction[1], 0]
            )[:2]
            v_fwd[idx+1] = np.dot(vel[idx+1], ht_direction)
            v_lat[idx+1] = np.dot(vel[idx+1], vec_left)

        return v_fwd[1:], v_lat[1:]

    else:
        return 0.0, 0.0


def compute_speed_PCA(links_positions, links_vel, sim_fraction=1.0):
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    n_steps            = links_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    links_pos_xy  = links_positions[-n_steps_considered:,:,:2]
    joints_vel_xy = links_vel[-n_steps_considered:,:,:2]
    time_idx      = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []

    for idx in range(time_idx):
        x = links_pos_xy[idx,:,0]
        y = links_pos_xy[idx,:,1]

        pheadtail = links_pos_xy[idx][0]-links_pos_xy[idx][-1] # head - tail
        vcom_xy   = np.mean( joints_vel_xy[idx], axis=0)

        covmat               = np.cov([x,y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index        = np.argmax(eig_values)
        largest_eig_vec      = eig_vecs[:, largest_index]

        ht_direction       = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec    = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0,0,1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return speed_forward, speed_lateral


# ------= SIGNAL PROCESSING TOOLS ------=
def filter_signals(
    signals   : np.ndarray,
    signal_dt : float,
    fcut_hp   : float = None,
    fcut_lp   : float = None,
    filt_order: int = 5,
):
    ''' Butterwort, zero-phase filtering '''

    # Nyquist frequency
    fnyq = 0.5 / signal_dt

    # Filters
    if fcut_hp is not None:
        num, den = butter(filt_order, fcut_hp/fnyq,  btype= 'highpass')
        signals  = filtfilt(num, den, signals)

    if fcut_lp is not None:
        num, den = butter(filt_order, fcut_lp/fnyq, btype= 'lowpass' )
        signals  = filtfilt(num, den, signals)

    return signals

def remove_signals_offset(signals: np.ndarray):
    ''' Removed offset from the signals '''
    return (signals.T - np.mean(signals, axis = 1)).T

def compute_frequency_treshold_crossings(
    times  : np.ndarray,
    signals: np.ndarray,
    theta = 0.1
    ):
    """
    Compute the frequency of a signal oscillating around a predefined treshold
    """

    onset_idxs = (signals[:,1:]>=theta) * (signals[:,:-1]<theta)
    last_spikes=np.zeros(onset_idxs.shape[0])
    prec_spikes=np.zeros(onset_idxs.shape[0])
    for idx, onsets in enumerate(onset_idxs):
        onsets = onset_idxs[idx]
        all_spikes = times[:-1][onsets]
        if len(all_spikes)>1 and all_spikes[-1]>times[-1]/2: # check if 1) all neurons have fired and 2) the last spike happened after tstop/2 (or no oscillations at st)
            last_spikes[idx] = all_spikes[-1]
            prec_spikes[idx] = all_spikes[-2]
    periods = last_spikes - prec_spikes

    if all(periods): # make sure that all neurons are oscillating
        period = np.mean(periods)
        phase_diffs  = np.zeros(onset_idxs.shape[0]-1)
        for i in range(len(last_spikes)-1):
            dphi = last_spikes[i+1]-last_spikes[i]
            if dphi<-period/2:
                phase_diffs[i] = period+dphi
            elif dphi>period/2:
                phase_diffs[i] = dphi-period
            else:
                phase_diffs[i] = dphi
        frequency = 1/period
    else:
        frequency  = 0.0
        phase_diffs = np.zeros(onset_idxs.shape[0])

    return frequency, phase_diffs

def compute_frequency_fft(
    times  : np.ndarray,
    signals: np.ndarray,
    )      :
    ''' Computes the average frequency based on the FFT '''

    dt_sig = times[1]-times[0]
    n_step = len(times)

    # Filter high-frequency noise
    smooth_signals = signals.copy()
    smooth_signals = remove_signals_offset(smooth_signals)
    smooth_signals = filter_signals(smooth_signals, dt_sig, fcut_lp=50)

    # Compute FFT
    signals_fft = np.fft.fft(smooth_signals, axis=1)
    freqs  = np.fft.fftfreq(n_step,d=dt_sig)

    signals_fft_mod  = np.abs(signals_fft)
    ind_max_signals  = np.argmax(signals_fft_mod[:, :n_step//2], axis= 1)
    freq_max_signals = freqs[ind_max_signals]
    amp_max_signals = 2*signals_fft_mod[:, :n_step//2][np.arange(signals_fft_mod.shape[0]),ind_max_signals]/signals_fft_mod.shape[1]

    return freq_max_signals, ind_max_signals, amp_max_signals


def compute_ptcc(
        times       : np.ndarray,
        signals     : np.ndarray,
        discard_time: float = 0,
        filtering   : bool  = True,
    ) -> np.ndarray:
    '''
    Computes the peak-to-through correlatio coefficient
    based on the difference between the the maximum and the minimum of the
    auto-correlogram of smoothened pools' oscillations
    '''

    sig_dt = times[1] - times[0]

    smooth_signals = signals.copy()
    if filtering:
        smooth_signals = filter_signals(smooth_signals, sig_dt, fcut_lp= 50)

    # Remove offset and initial transient
    smooth_signals = remove_signals_offset(smooth_signals)

    istart = round( float(discard_time)/float(sig_dt) )
    chopped_signals = smooth_signals[:, istart:]

    ptccs = []
    for ind, sig in enumerate(chopped_signals):

        # Auto-correlogram
        power    = np.correlate(sig, sig)
        autocorr = fftconvolve(sig, sig[::-1], 'full')

        if power > 0:
            autocorr = autocorr / power
        else:
            autocorr = np.zeros(autocorr.shape)

        siglen = len(autocorr)//2   # Computed in [-n//2,+n//2-1]

        sig_maxs = find_peaks(+autocorr[siglen:])
        sig_mins = find_peaks(-autocorr[siglen:])

        if sig_maxs[0].size == 0 or sig_mins[0].size == 0:
            ptccs.append(0)
        else:
            first_max_ind = sig_maxs[0][0] + siglen
            first_min_ind = sig_mins[0][0] + siglen

            ptccs.append(autocorr[first_max_ind] - autocorr[first_min_ind])


    return np.array(ptccs)



def compute_controller(
        network
):
    """
    Compute all the metrics of the firing rate controller
    """
    metrics = {}

    n_iterations      = network.pars.n_iterations
    sim_fraction      = 0.6
    nsteps_considered = round(n_iterations * sim_fraction)
    exit_iteration    = network.exit_iteration

    # consider sim_fraction number of steps, for the difference between left and right muscle cells
    times   = network.times[-nsteps_considered:exit_iteration]

    njoints_total = 15
    idxs = range(8) # 10 considered muscle pairs (no muscles for the first 5 joints)

    signals = network.state[-nsteps_considered:exit_iteration,network.muscle_r[idxs]]-network.state[-nsteps_considered:exit_iteration,network.muscle_l[idxs]]
    fraction_considered = njoints_total/len(idxs)

    frequency, iplss = compute_frequency_treshold_crossings(
        times, signals.transpose(), theta=0.001
    )

    iplss = np.append(iplss,iplss[-1]) # adding one element to consider the full body length
    metrics["frequency"]     = frequency
    metrics["ipls"]          = np.sum(iplss)*fraction_considered # correction to consider number of considered joints (-discarted)
    metrics["wavefrequency"] = metrics["ipls"]*metrics["frequency"]

    ptcc = compute_ptcc(
            times,
            signals.transpose(),
            )
    metrics["ptcc"] = np.min( ptcc )

    # ------ amplitude of the limbs at steady-state ------
    metrics["amp"] = np.max(signals)-np.min(signals)


    return metrics

def compute_mechanical(
    network
):
    """
    compute all mechanical metrics
    """


    metrics = {}
    sim_fraction = 0.6


    network.exit_iteration=3300
    # ------ COMPUTE FORWARD AND LATERAL SPEEDS ------
    links_positions       = network.links_positions[:network.exit_iteration]
    link_velocities       = network.links_velocities[:network.exit_iteration]
    joints_active_torques = network.joints_active_torques[:network.exit_iteration]

    # method 1
    (fspeed, lspeed) = compute_speed_PCA(
        links_positions,
        link_velocities,
        sim_fraction = sim_fraction
        )

    metrics["fspeed_PCA"] = np.median(fspeed/BODY_LENGTH)
    metrics["lspeed_PCA"] = np.median(lspeed/BODY_LENGTH)

    # method 2
    (fspeed, lspeed) = compute_speed_cycle(
        network,
        sim_fraction = sim_fraction
    )
    metrics["fspeed_cycle"] = np.mean(fspeed)
    metrics["lspeed_cycle"] = np.mean(lspeed)

    # ------ Mechanical metrics ------
    torque   = sum_torques(joints_active_torques, sim_fraction=sim_fraction)

    metrics["torque"]   = torque

    curvatures = compute_curvature(
                network,
                sim_fraction = sim_fraction
    )
    metrics["curvature"] = np.mean(curvatures)

    if network.mujoco_error==True:
        metrics = dict.fromkeys(metrics, np.nan)

    return metrics


def compute_all(network):
    """
    compute and joint all controller and mechanical metrics
    """

    if sys.version_info[1] <= 8: # for python3.8 or lower
        metrics = {**compute_controller(network), **compute_mechanical(network)}
    else:
        metrics = compute_controller(network) | compute_mechanical(network)
    return metrics
