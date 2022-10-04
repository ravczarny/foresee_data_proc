# spectral whitening
import numpy as np
import scipy.fftpack


def spec_whiten(data):

    lenght = len(data)
    nfft = int(2 ** np.ceil(np.log2(np.abs(lenght))))
    white_trace = np.exp(1j * np.angle(scipy.fftpack.fft(data, nfft)))
    white_trace = np.real((scipy.fftpack.ifft(white_trace)))
    white_trace = white_trace[:len(data)]
    return white_trace


def spec_whiten_parallel(data):
    # Function normalizes amplitude spectrum in frequency domain i.e. makes it more uniform
    # basic eq:
    # Output_signal(omega) = |input_signal(omega)|.*1/|input_signal(omega)|./exp(1j*angle(fft(input_signal)))
    #
    # Author : Rafał Czarny
    #
    # -------------------------------input----------------------------------
    #
    #   data : seismic signal array [samples,traces]
    #
    lenght = data.shape[0]
    nfft = int(2 ** np.ceil(np.log2(np.abs(lenght))))
    white_tmp = np.exp(1j * np.angle(scipy.fftpack.fft(data, nfft, axis=0)))
    white_tmp = np.real((scipy.fftpack.ifft(white_tmp, axis=0)))
    white_tmp = white_tmp[:lenght, :]
    return white_tmp
