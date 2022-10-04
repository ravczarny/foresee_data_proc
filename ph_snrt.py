"""
determine SNR
author Rafal Czarny rkc5556@psu.edu
"""
import numpy as np
from os import listdir, scandir
from os.path import isfile, join
import ray
from scipy import fft, ifft
from phase_shift import phase_shift_1d

ccfs_input = '/egg1/s1/rkc5556/projects/foresee/dvv/output/sw_PL_1460_1860/vs1665_1760/negative/2019_05/'
ccf_output = '/egg1/s1/rkc5556/projects/foresee/dvv/output/sw_PL_may_selected_hi_snr/negative/2019_05/'

for iday in range(1, 32):  # day

    if iday < 10:
        day_name = str(0) + str(iday)
    else:
        day_name = str(iday)
    ray.shutdown()


    @ray.remote
    def screening(file):
        res = []
        for vs in range(19):
            # print(vs)
            tmp = np.load(file, allow_pickle=True)[0][vs]
            for ch in range(tmp.shape[0]):
                tmp[ch, :] = tmp[ch, :] / max(tmp[ch, :])
            slow_n, output = phase_shift_1d(tmp, 0.01, 2, 0, (tmp.shape[0]) * 2, 'negative')
            output_f0 = output[-16:]
            output_noise = output[:-17]
            amax = np.where(output_f0 == np.max(output_f0))
            snr = output_f0[amax[0]] / np.mean(output_noise)

            res.append([snr[0], file])
        return res

    ray.init(num_cpus=80)
    files = [ccfs_input + day_name + '/' + f for f in listdir(ccfs_input + day_name + '/')]

    result = []
    for d, file in enumerate(files):
        print(file)

        z1 = screening.remote(file)
        result.append(z1)
    results = ray.get(result)
    for item in results:
        for virt in range(len(item)):
            # if virt < 10:
            #     vs_name = str(0) + str(virt)
            # else:
            #     vs_name = str(virt)
            vs_name = str(virt + 39)
            # save SNR for single VSGs
            with open(ccf_output + day_name + '/' + vs_name + '/' + 'snr_1', 'a') as f:
                f.write(str(item[virt][0][0]) + ' ' + str(str(item[virt][1])) + '\n')
    del results

    #
