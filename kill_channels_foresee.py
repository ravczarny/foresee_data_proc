"""
remove channels
Author: Rafal Czarny rkc5556@psu.edu
"""
import numpy as np


# form MATLAB - remember -1 applied due to ordering in python
def kill_foresee(idas_array):
    bad_traces = list(range(0, 23)) \
                 + list(range(81, 88)) \
                 + list(range(144, 155)) \
                 + list(range(214, 220)) \
                 + list(range(365, 367)) \
                 + list(range(428, 435)) \
                 + list(range(611, 618)) \
                 + list(range(672, 683)) \
                 + list(range(1017, 1032)) \
                 + list(range(1050, 1073)) \
                 + list(range(1094, 1098)) \
                 + list(range(1130, 1134)) \
                 + list(range(1174, 1176)) \
                 + list(range(1219, 1222)) \
                 + list(range(1224, 1227)) \
                 + list(range(1234, 1239)) \
                 + list(range(1244, 1255)) \
                 + list(range(1264, 1272)) \
                 + list(range(1530, 1587)) \
                 + list(range(2084, 2096)) \
                 + list(range(2119, 2126)) \
                 + list(range(2365, idas_array.shape[1]))
    idas_array = np.delete(idas_array, bad_traces, 1)
    return idas_array
