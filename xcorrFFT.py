"""
Cross-correlation
Author: Rafal Czarny rkc5556@psu.edu
"""

import numpy as np
import scipy.fftpack


def xcorrFFT_paralell_single(data, lagtime, fs, first_channel, last_channel):
    import scipy.fftpack
    """
    :param data: 2D 
    :param lagtime: time window fo 0-lag time in (sec)
    :param fs: frequency sampling
    :v_source : channel for virtual source
    """
    # divide data into 30 sec

    fs = int(fs)
    lag_samples = 2 * lagtime * fs + 1

    cc_vs = []
    for v_source in range(10, 200, 20):  # range(10, 270, 20):
        print(v_source, 'v_source')

        Nt = data.shape[0]
        Nc = 2 * Nt - 1
        Nfft = 2 ** np.ceil(np.log2(np.abs(Nc)))
        fft_array = scipy.fftpack.fft(data.T, int(Nfft), axis=-1)
        if Nt > lag_samples:
            # cross correlate all array for 1 virtual source in v_source
            ccfs = np.empty([0, lag_samples])
            for ii in range(first_channel, last_channel, 1):
                cross_temp1 = np.conj(fft_array[v_source]) * fft_array[ii]
                cross_temp2 = np.real(scipy.fftpack.ifft(cross_temp1))
                cross_temp3 = np.concatenate((cross_temp2[-Nt + 1:], cross_temp2[:Nt]))
                cross_temp4 = cross_temp3[(data.shape[0] - lagtime * fs - 1):(data.shape[0] + lagtime * fs)]
                ccfs = np.append(ccfs, [cross_temp4.T], axis=0)
        else:
            pass
        cc_vs.append(ccfs)
    return cc_vs

