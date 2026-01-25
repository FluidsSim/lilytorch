
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, butter, filtfilt
import pywt
from scipy import ndimage
import os
from ssqueezepy import Wavelet, cwt

def get_schooling_data(
    times       : np.ndarray,
    joints_pos_1: np.ndarray,
    joints_pos_2: np.ndarray,
    links_pos_1 : np.ndarray,
    links_pos_2 : np.ndarray,
    plotting    : bool = True,
    data_dir    : str = None,
    freq0       : float = None,
    freq1       : float = None,
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
    angle_0 = np.sum(joints_pos_1, axis=1)
    angle_1 = np.sum(joints_pos_2, axis=1)

    # Get com position evolution
    com_pos_0 = np.mean(links_pos_1, axis=1)
    com_pos_1 = np.mean(links_pos_2, axis=1)

    com_posx_0 = com_pos_0[:, 0]
    com_posx_1 = com_pos_1[:, 0]

    fs = 1 / timestep
    fmin = 0.4          # Minimum frequency of interest (Hz)
    fmax = 1.5         # Maximum frequency of interest (Hz)
    duration = times[-1] - times[0]
    N=int(fs * duration)
    wavelet = Wavelet(('morlet', {'mu': 5}), N=N)
    wc_ct = wavelet.wc_ct
    min_scale = (wc_ct / (2*np.pi)) * (fs / fmax)   # corresponds to highest freq
    max_scale = (wc_ct / (2*np.pi)) * (fs / fmin)   # corresponds to lowest freq
    nv = 32
    scales = np.logspace(np.log2(min_scale), np.log2(max_scale), int(np.ceil(nv*np.log2(max_scale/min_scale))), base=2)
    scales = scales.reshape(-1,1)
    def compute_cwt(signal):
        Wx, scales_out = cwt(signal, wavelet=wavelet, scales=scales, fs=fs)
    #     S = np.abs(W**2)/scaleMatrix
    #     if smoothing:
    #         S = ndimage.gaussian_filter(S, sigma=sigma)

    #     freq_idx = np.argmax(S,axis=0)
    #     power_1d = S[freq_idx,range(n)]
    #     # freq_max = np.where(power_1d>0.1, frequencies[freq_idx], 0)
    #     freq_max = frequencies[freq_idx]
    #     return W, freq_max, power_1d

    # n         = angle_1.shape[0]
    # normalize = True
    # smoothing = True
    # sigma     = 2

    # freqs       = np.geomspace(1, 0.6, num=1000)
    # wavelet     = 'cmor1.0-3.0'
    # scales      = 1 / (freqs*timestep)
    # frequencies = pywt.scale2frequency(wavelet, scales) / timestep
    # scaleMatrix = np.ones([1, n]) * scales[:, None]

    # # step 1: compute the cwt and maxial power frequency of the normalized signals
    # def compute_cwt(signal):
    #     if normalize:
    #         signal = (signal - signal.mean()) / signal.std()
    #     [W, _] = pywt.cwt(
    #         signal,
    #         scales,
    #         wavelet,
    #         timestep,
    #         method="fft"
    #         )
    #     S = np.abs(W**2)/scaleMatrix
    #     if smoothing:
    #         S = ndimage.gaussian_filter(S, sigma=sigma)

    #     freq_idx = np.argmax(S,axis=0)
    #     power_1d = S[freq_idx,range(n)]
    #     # freq_max = np.where(power_1d>0.1, frequencies[freq_idx], 0)
    #     freq_max = frequencies[freq_idx]
    #     return W, freq_max, power_1d

    # # step 2: compute cross-wavelet transform and the maximal power phase difference angle
    # def compute_cross_coherence(W1, W2, smoothing=smoothing):
    #     xwt       = W1 * W2.conj()
    #     power_xwt = np.abs(xwt**2)/scaleMatrix
    #     if smoothing:
    #         power_xwt = ndimage.gaussian_filter(power_xwt, sigma=sigma)
    #         # power_xwt = ndimage.gaussian_filter1d(power_xwt, sigma, 1)
    #     phase_xwt = np.angle(xwt)
    #     freq_idx  = np.argmax(power_xwt,axis=0)
    #     phi_max   = phase_xwt[freq_idx,range(n) ]
    #     return phi_max


    # W_0, freq_0, power_0 = compute_cwt(angle_0)
    # W_1, freq_1, power_1 = compute_cwt(angle_1)
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # extent = [times[0], times[-1], frequencies[0], frequencies[-1]]

    # im0 = axes[0].imshow(
    #     np.abs(W_0**2)/scaleMatrix,
    #     aspect='auto',
    #     origin='lower',
    #     extent=[times[0], times[-1], frequencies[0], frequencies[-1]]
    # )
    # axes[0].set_title('CWT Magnitude - Agent 1')
    # axes[0].set_xlabel('Time (s)')
    # axes[0].set_ylabel('Frequency (Hz)')
    # fig.colorbar(im0, ax=axes[0], label='Power')

    # im1 = axes[1].imshow(
    #     np.abs(W_1**2)/scaleMatrix,
    #     aspect='auto',
    #     origin='lower',
    #     extent=[times[0], times[-1], frequencies[0], frequencies[-1]]
    # )
    # axes[1].set_title('CWT Magnitude - Agent 2')
    # axes[1].set_xlabel('Time (s)')
    # axes[1].set_ylabel('Frequency (Hz)')
    # fig.colorbar(im1, ax=axes[1], label='Power')

    # plt.tight_layout()
    # plt.savefig(os.path.join(data_dir, "cwt_agents_" + str(freq1) + ".png"))
    # plt.close(fig)

    # phases = compute_cross_coherence(W_0, W_1)

    # if plotting:
    #     fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    #     # Plot freq_1 and freq_2
    #     axes[0].plot(times, freq_0, label='Agent 1 (computed)', color='blue', linewidth=1.5)
    #     axes[0].plot(times, freq_1, label='Agent 2 (computed)', color='red', linewidth=1.5)
    #     if freq0 is not None:
    #         axes[0].plot(times, np.full_like(times, freq0), 'b:', label='Agent 1 (actual)', linewidth=1.2)
    #     if freq1 is not None:
    #         axes[0].plot(times, np.full_like(times, freq1), 'r:', label='Agent 2 (actual)', linewidth=1.2)
    #     axes[0].set_ylabel('Frequency (Hz)')
    #     axes[0].legend()
    #     axes[0].grid(True, alpha=0.3)

    #     # Plot angle_1 and angle_2
    #     axes[1].plot(times, angle_0, label='Agent 1', color='blue', linewidth=1.5)
    #     axes[1].plot(times, angle_1, label='Agent 2', color='red', linewidth=1.5)
    #     axes[1].set_ylabel('Angle (rad)')
    #     axes[1].set_title('Joint Angles')
    #     axes[1].legend()
    #     axes[1].grid(True, alpha=0.3)

    #     # Plot phase difference
    #     axes[2].plot(times, phases, label='Phase Difference', color='green', linewidth=1.5)
    #     axes[2].set_ylabel('Phase (rad)')
    #     axes[2].legend()
    #     axes[2].grid(True, alpha=0.3)
    #     axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(data_dir, "phase_analysis_schooling_" + str(freq1) + ".png"))
    #     plt.close(fig)



    return times, freq_0, freq_1, phases



