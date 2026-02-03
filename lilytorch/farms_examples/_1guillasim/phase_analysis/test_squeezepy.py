import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import cwt, ssq_cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq
from scipy.signal import chirp

# ============================================================
# Parameters
# ============================================================
fs = 500         # Sampling frequency (Hz)
duration = 10.0     # Signal duration (s)

fmin = 0.4          # Minimum frequency of interest (Hz)
fmax = 1.5         # Maximum frequency of interest (Hz)

N=int(fs * duration)
# ============================================================
# Time vector
# ============================================================
t = np.linspace(0, duration, N, endpoint=False)

# ============================================================
# Example signal: quadratic chirp (1 Hz -> 30 Hz)
# ============================================================
# Create a quadratic chirp from fmin to fmax
k = (fmax - fmin) / (duration ** 2)
signal = chirp(t, f0=fmin, t1=duration, f1=fmax, method='quadratic')




# #%%# With units #######################################
from ssqueezepy import Wavelet, cwt, imshow
# choose Morlet with a different central-frequency parameter (e.g. mu=8)
wavelet = Wavelet(('morlet', {'mu': 5}), N=N)

# Wx, scales = cwt(signal, wavelet)
# freqs_cwt = scale_to_freq(scales, wavelet, len(signal), fs=fs)
# ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
# imshow(Wx, **ikw, yticks=freqs_cwt)



wc_ct = wavelet.wc_ct
min_scale = (wc_ct / (2*np.pi)) * (fs / fmax)   # corresponds to highest freq
max_scale = (wc_ct / (2*np.pi)) * (fs / fmin)   # corresponds to lowest freq
nv = 32
scales = np.logspace(np.log2(min_scale), np.log2(max_scale), int(np.ceil(nv*np.log2(max_scale/min_scale))), base=2)
scales = scales.reshape(-1,1)
Wx, scales_out = cwt(signal, wavelet=wavelet, scales=scales, fs=fs)

freqs_cwt = scale_to_freq(scales, wavelet, len(signal), fs=fs)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, signal)
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.title("Signal")
plt.grid(True)

plt.subplot(2,1,2)
ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
imshow(Wx, **ikw, yticks=freqs_cwt)

S = np.abs(Wx**2)
freq_idx = np.argmax(S,axis=0)
freq_max = freqs_cwt[freq_idx]
power_1d = S[freq_idx,range(signal.shape[0])]

plt.figure(figsize=(10,4))
plt.plot(t, freq_max)
plt.xlabel("Time [sec]")
plt.ylabel("Frequency [Hz]")
plt.title("Computed Instantaneous Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
