import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft, next_fast_len


def pws(arr, sampling_rate, power=2, pws_timegate=5.):
    '''
    Schimmel and Paulssen, 1997
    '''

    if arr.ndim == 1:
        return arr
    N, M = arr.shape
    analytic = hilbert(arr, axis=1, N=next_fast_len(M))[:, :M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j * phase), axis=0)
    phase_stack = np.abs(phase_stack) ** (power)
    weighted = np.multiply(arr, phase_stack)
    return np.mean(weighted, axis=0)
