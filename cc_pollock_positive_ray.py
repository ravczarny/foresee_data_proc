"""
ambient noise interferometry workflow / positive wavefield
Author: Rafal Czarny rkc5556@psu.edu
"""
import os
import time
from xcorrFFT import xcorrFFT_paralell_single
import numpy as np
from filters import *
import ray
from scipy import signal
from spec_whiten import spec_whiten_parallel
import scipy as sp
import scipy.ndimage
from fk_filtering import fk_filt_cc, fk_filt_source_cc

das_input = '/mnt/DAS/may20/'
filename_ending = '.tdms'
ccf_output = '/egg1/s1/rkc5556/projects/foresee/dvv/output/sw_PL_selected/negative/2020_05/'

cpus = 85  # CPUs
decimation_factor = 5


# --------------------------    PROC 1a
@ray.remote
def load_das_files(file_path):
    from tdms_reader import TdmsReader
    from kill_channels_foresee import kill_foresee
    idas_file = TdmsReader(file_path)
    n_samples = idas_file.channel_length
    n_channels = idas_file.fileinfo['n_channels']
    data = idas_file.get_data(0, n_channels, 0, n_samples)
    data = kill_foresee(data)
    props = idas_file.get_properties()
    fs = props.get('SamplingFrequency[Hz]')
    data1 = data, fs
    return data1


# --------------------------    PROC 2
# detrend
@ray.remote
def das_detrend(data):
    print('detrend', data[0].shape, data[1])
    data1 = signal.detrend(data[0], axis=0, type='linear', bp=5000)
    data2 = data1, data[1]
    print('detrend', data2[0].shape, data2[1])
    return data2

# --------------------------    PROC 3
# decimation
@ray.remote
def decimate_das(data):
    print('decimated', data[0].shape, data[1])
    data1 = signal.decimate(data[0], decimation_factor, axis=0, ftype='fir')
    data2 = data1, int(data[1] / decimation_factor)
    print('decimation', data2[0].shape, data2[1])
    return data2  # columns are traces

# --------------------------    PROC 3
# fk
@ray.remote
def fk(data):
    print('fk', data[0].shape, data[1])
    # ---- one mask for all
    # mask = np.load('fk_mask/fk_mask_negative_200_6000_100hz.npy')
    mask = np.load('fk_mask/fk_mask_positive_200_6000_100hz.npy')

    data1 = fk_filt_cc(data[0], mask)
    data2 = data1, data[1]
    print('fk', data2[0].shape, data2[1])
    return data2  # columns are traces


# --------------------------    PROC 4
# bandpass filtering _band pass
@ray.remote
def das_filtering(data):
    print('bandpass', data[0].shape, data[1])
    data1 = butter_bandpass_filter_2d(data[0], 1, 49, data[1])
    data2 = data1, data[1]
    print('bandpass', data2[0].shape)
    return data2


# --------------------------    PROC 5
# amplitude correction / spectral whitening
@ray.remote
def das_whiten(data):
    print('whiten', data[0].shape, data[1])
    data1 = spec_whiten_parallel(data[0])
    data2 = data1, data[1]
    print('whiten', data2[0].shape)
    return data2


# --------------------------    PROC 6
# correlation
@ray.remote
def corr(data):
    print('corr', data[0].shape, data[1])
    lag_time = 5
    data1 = xcorrFFT_paralell_single(data[0], lag_time, data[1], 1460, 1790)  # 1460,1860 before
    data2 = data1, data[1]
    print('corr', )
    return data2


# read files in dir
def files_list(path):
    f = os.listdir(path)
    f = sorted(f)
    return f

# read all files in DIR
files = os.listdir(das_input)
files = sorted(files)
files_day = []  # use only files during a day
for file in files:
    if 12 < int(file[20:22]) < 22:
        files_day.append(file)

# files_day = files_day[:20000]
# das_files_all = {os.path.join(das_input, fp) for fp in os.listdir(f"{das_input}") if fp.endswith(filename_ending)}
das_files_all = {das_input + f for f in files_day}

# remove processed files from das_files basket
ccf_files_basket = files_list(ccf_output)
ccf_files_basket = [das_input + ccf_files_basket[x][:-4] + filename_ending for x in range(len(ccf_files_basket))]
print(len(ccf_files_basket), len(das_files_all))
das_files = list(das_files_all.difference(ccf_files_basket))
print(len(das_files), 'dd')
ray.init(num_cpus=cpus, _plasma_directory="/egg1/s1/rkc5556/data/plasma_ray/", object_store_memory=10**9)
tic = time.clock()
for rd in range(1000):
    files_day = das_files[rd * cpus:(rd + 1) * cpus - 1]
    ccf_files_basket = files_list(ccf_output)
    print('Total number of files is: {0}'.format(len(das_files_all)), ' and processed {0}'.format(len(ccf_files_basket)))

    result = []
    # processing workflow
    for i in range(len(files_day)):
        z1 = load_das_files.remote(files_day[i])
        z2 = das_detrend.remote(z1)
        z3 = decimate_das.remote(z2)
        z3b = fk.remote(z3)
        z4 = das_filtering.remote(z3b)
        z5 = das_whiten.remote(z4)
        z6 = corr.remote(z5)
        result.append(z6)
    results = ray.get(result)
    for id in range(len(files_day)):
        np.save(ccf_output + files_day[id][-35:-5], np.array(results[id]))
    del results
toc = time.clock()
print(toc - tic)
