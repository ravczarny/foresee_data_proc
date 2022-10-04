"""
phase-shift / slant stack
Author: Rafal Czarny rkc5556@psu.edu
"""
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def phase_shift_1d(data, dt, dx, ch_start, ch_end, wavenumber):

    # data = data[:,300:700]
    if wavenumber == 'positive':
        slow_n = np.arange(0.25, 1, 0.02)  # (s/km)
    if wavenumber == 'negative':
        slow_n = -np.arange(0.25, 1, 0.02)  # (s/km)
    offset = np.arange(ch_start / 1000, ch_end / 1000, dx / 1000)  # offset (km)

    for ch in range(data.shape[0]):
        data[ch, :] = data[ch, :] / max(data[ch, :])

    omega = np.arange(data.shape[1]) / (dt * data.shape[1]) * 2 * np.pi

    output = np.zeros((len(slow_n), 1))
    for i, slowness in enumerate(slow_n):
        tmp0 = fft(data, axis=1) * np.exp(1j * np.outer(offset * slowness, omega))
        tmp1 = np.real(ifft(tmp0, axis=1))
        output[i, :] = max(abs(sum(tmp1)))

    return slow_n, output

