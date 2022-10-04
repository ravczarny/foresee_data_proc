"""
pws stacking
Author: Rafal Czarny rkc5556@psu.edu
"""
import numpy as np
from os import listdir, scandir
from os.path import isfile, join
import ray
from scipy import fft, ifft
from phase_shift import beamform_1d

ccfs_input = '/egg1/s1/rkc5556/projects/foresee/dvv/output/sw_PL_1460_1860/vs1565_1660/negative/2019_05/'
ccf_output = '/egg1/s1/rkc5556/projects/foresee/dvv/output/sw_PL_1460_1860/vs1565_1660/slant_stack_75/'

cpus = 35 # CPUs
# virual_source = 0

for virual_source in range(0,16,1):
    ray.shutdown()
    # --------------------------    PROC 1  ----------------------------------------
    @ray.remote
    def file_database(dir_path):
        files = [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        return files


    # --------------------------    PROC 2  ----------------------------------------
    @ray.remote    # Function normalizes amplitude spectrum in frequency domain i.e. makes it more uniform
    # basic eq:
    # Output_signal(omega) = |input_signal(omega)|.*1/|input_signal(omega)|./exp(1j*angle(fft(input_signal)))
    #
    # Author : Rafał Czarny
    #
    # -------------------------------input----------------------------------
    #
    #   data : seismic signal
    #
    def stack(files):
        count = 0
        total = 0
        ccfs = np.zeros((400, 1001))
        for i, file in enumerate(files):


            tmp = np.load(file, allow_pickle=True)[0][virual_source]
            for ch in range(tmp.shape[0]):
                tmp[ch, :] = tmp[ch, :] / max(tmp[ch, :])
            slow_n, output = beamform_1d(tmp, 0.01, 2, 0, (tmp.shape[0]) * 2, 'negative')
            output_f0 = output[-16:]
            output_noise = output[:-17]
            amax = np.where(output_f0 == np.max(output_f0))
            snr = output_f0[amax[0]] / np.mean(output_noise)

            if snr > 2.0:
                if count < 75:
                    ccfs += tmp
                    total +=1
                count += 1
                # print(count, i)
                # ccfs += tmp
        # res = ccfs
        res = [ccfs, total]
        return res


    days_path = [ccfs_input + f + '/' for f in listdir(ccfs_input)]
    ray.init(num_cpus=cpus)  # , _plasma_directory="/egg1/s1/rkc5556/data/plasma_ray/")  # , object_store_memory=10**9)


    days = days_path#[rd * cpus:(rd + 1) * cpus - 1]  # select days the same as CPUs for mpi

    result = []
    for d in range(len(days)):
        print(days[d])
        z1 = file_database.remote(days[d])
        z2 = stack.remote(z1)
        result.append(z2)
    results = ray.get(result)
    for id in range(len(days)):
        np.save(ccf_output+ccfs_input[-8:-1] + days[id][-3:-1] + '_' + str(virual_source+19) + 'n_' + str(results[id][1]),
                results[id][0])
    del results


