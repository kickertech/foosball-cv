import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import peakutils

import matplotlib.pyplot as plt

def find_peaks(frame, target_color, rodCandidate, sigma=5, threshold=0.4, step_size=3):
    (y0, y1) = rodCandidate
    yit = np.arange(y0, y1)
    ySlice = frame[yit]
    (sh, sw) = ySlice.shape[0:2]
    if sw % step_size != 0:
        raise Exception("step_size must be divisor of frame width")

    tc = np.zeros(((sw*sh)//step_size, 3))
    tc[:][:] = target_color

    # get mean for every 3element group
    group_mean = np.mean(
        ySlice.reshape((sw*sh)//step_size, step_size, 3), axis=1)

    # zip mean values and target-color
    mst = np.dstack((group_mean, tc))

    # get abs diff
    ad = np.absolute(np.diff(mst, axis=2))

    # average over color channels
    ad = np.mean(ad, axis=1)

    # average over cols
    rad = np.mean(ad.reshape(sh, sw//step_size), axis=0)

    # repeat values to get full width again
    avgXDiff = np.repeat(rad, step_size, axis=0)
    rowBlurred = gaussian_filter(avgXDiff, sigma=sigma)

    # DEBUG: plot distribution
    # fig, ax = plt.subplots()
    # ax.plot(rowBlurred / max(rowBlurred), label='team 2')
    # plt.legend()
    # ax.set(xlabel='sigma {} / y: {}-{}'.format(sigma, y0, y1), ylabel='difference')
    # ax.grid()
    # plt.show()

    return peakutils.indexes(rowBlurred, thres=threshold, min_dist=sw//30)
