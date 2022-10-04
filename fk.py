"""
fk filtering
Author: Rafal Czarny rkc5556@psu.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fftn, ifftshift, ifftn
import scipy as sp
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def fk_filt(data, dt, dx, v_min, v_max, sigma=None, plots=False):
    """
    :param data: 2D DAS array
    :param dt: sampling in sec
    :param dx: channel spacing in meters
    :param v_min: minimum phase velocity to pass
    :param v_max: maximum phase velocity to pass
    :param sigma: gaussian filter shape (rows,cols)
    :param plots: True if you want to see the results
    :return: filtered data in same shape as input
    """
    Nk = int(2 ** np.ceil(np.log2(np.abs(data.shape[1]))))
    Nf = int(2 ** np.ceil(np.log2(np.abs(data.shape[0]))))
    print(Nk, Nf)
    k_sc = (1 / dx) * np.arange(-Nk / 2, Nk / 2 + 1) / Nk
    fr_sc = (1 / dt) * np.arange(-Nf / 2, Nf / 2 + 1) / Nf
    fk = fftshift(np.fft.fft2(data, s=(Nf, Nk)))
    fk_plt = np.copy(fk)
    # ----------  Design the mask shape  ---------- #
    xk, yf = np.meshgrid(k_sc, fr_sc)
    for i, k in enumerate(k_sc):
        yf[:, i] = yf[:, i] / (k + 1e-10)
    yf_positive = np.where(np.logical_and(yf >= -v_max, yf <= -v_min), yf, 0)
    yf_negative = np.where(np.logical_and(yf >= v_min, yf <= v_max), yf, 0)
    mask = yf_negative + yf_positive
    mask = np.where(mask == 0, mask, 1)
    # Apply gaussian filter to smooth mask
    if sigma is None:
        sigma = [10, 10]
    mask = sp.ndimage.filters.gaussian_filter(mask, sigma, mode='reflect')
    # np.save('fk_mask_negative_200_6000_100hz', mask) # save mask for fk_filt_cc
    # ----------  filter  ---------- #
    fk *= mask[:-1, :-1]
    m = ifftshift(fk)
    data_flt = np.fft.ifft2(m).real
    data_flt = data_flt[:data.shape[0], :data.shape[1]]
    upper = mpl.cm.turbo(np.arange(256))
    lower = np.ones((int(256 / 4), 4))
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

    # combine parts of colormap
    cmap = np.vstack((lower, upper))

    # convert to matplotlib colormap
    cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

    if plots:

        f1, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharey=True, sharex=True)
        extent = (k_sc[0], k_sc[-1], fr_sc[0], fr_sc[-1])

        cb = ax[0].imshow((np.abs(fk_plt)), aspect='auto', extent=extent, cmap=cmap, vmax=np.max(np.abs(fk)) * 0.5)
        for i, vel in enumerate([100, 300, 500, 1000, 2000, 4000]):
            ax[0].plot([-fr_sc[0] / vel, fr_sc[0] / vel], [-fr_sc[0], fr_sc[0]], '--w', linewidth=.5)
            t = ax[0].text((-i - 1) * 30 / vel, (-i - 1) * 30, str(vel) + ' m/s', fontsize=8)
            t.set_bbox(dict(facecolor='w', alpha=.5))
        ax[0].set_title('f-k before')
        # plt.colorbar(cb)
        ax[0].set_ylim([0, -100])
        ax[0].set_xlim([-0.1, 0.1])

        ax[1].imshow(mask, aspect='auto', extent=extent, cmap=cmap)

        ax[1].set_title('Mask')
        for i, vel in enumerate([v_min, v_max]):
            ax[1].plot([-fr_sc[0] / vel, fr_sc[0] / vel], [-fr_sc[0], fr_sc[0]], '--w', linewidth=.5)
            t = ax[1].text((-i - 1) * 30 / vel, (-i - 1) * 30, str(vel) + ' m/s', fontsize=8)
            t.set_bbox(dict(facecolor='w', alpha=.5))

        ax[2].imshow((np.abs(fk)), aspect='auto', extent=extent, cmap=cmap,
                     vmax=np.max(np.abs(fk)) * 0.5)  # [::10, ::10]
        for i, vel in enumerate([v_min, v_max]):
            ax[2].plot([-fr_sc[0] / vel, fr_sc[0] / vel], [-fr_sc[0], fr_sc[0]], '--w', linewidth=.5)
            t = ax[2].text((-i - 1) * 30 / vel, (-i - 1) * 30, str(vel) + ' m/s', fontsize=8)
            t.set_bbox(dict(facecolor='w', alpha=.5))

        ax[2].set_title('f-k after')
        plt.setp(ax, xlabel=r'Wavenumber/$ \pi $ (1/m)', ylabel='Frequency (Hz)')
        plt.setp(ax, xlim=[k_sc[0], k_sc[-1]], ylim=[0, fr_sc[0]])
        ax[2].set_ylim([0, -100])
        ax[2].set_xlim([-0.1, 0.1])
        plt.tight_layout()
        plt.savefig('grl_s1_fk_negative.eps', dpi=300, format='eps')
        plt.show()
        # figure 2 - data before and after filtering
        f2, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, sharex=True)
        d_mean = np.mean(data), np.mean(data_flt)
        d_sdev = np.std(data), np.std(data_flt)
        c_max = d_mean[0] + 1.5 * d_sdev[0], d_mean[1] + 1.5 * d_sdev[1]
        c_min = d_mean[0] - 1.5 * d_sdev[0], d_mean[1] - 1.5 * d_sdev[1]
        extent = (0, data.shape[1], dt * data.shape[0], 0)
        ax[0].imshow(data, aspect='auto', vmin=c_min[0], vmax=c_max[0], extent=extent, cmap='gray')
        ax[0].set_title('Before')
        ax[1].imshow(data_flt, aspect='auto', vmin=c_min[0], vmax=c_max[0], extent=extent, cmap='gray')
        ax[1].set_title('After')
        plt.setp(ax, xlabel='Channel', ylabel='Time (s)')
        plt.tight_layout()
        plt.show()

    return data_flt


def fk_filt_cc(data, mask):
    """
    filtering with mask from fk_filt
    :param data: 2D DAS array
    :param mask: mask from outer source file
    :return: filtered data in same shape as input
    """
    Nk = int(2 ** np.ceil(np.log2(np.abs(data.shape[1]))))
    Nf = int(2 ** np.ceil(np.log2(np.abs(data.shape[0]))))
    fk = fftshift(np.fft.fft2(data, s=(Nf, Nk)))
    # ----------  filter  ---------- #
    fk *= mask[:-1, :-1]
    m = ifftshift(fk)
    data_flt = np.fft.ifft2(m).real
    data_flt = data_flt[:data.shape[0], :data.shape[1]]
    return data_flt
