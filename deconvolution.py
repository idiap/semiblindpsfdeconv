'''
Code for the pytorch implementation of
"Semi-Blind Spatially-Variant Deconvolution in Optical Microscopy
with Local Point Spread Function Estimation By Use Of Convolutional Neural Networks"

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,

This file is part of Semi-blind Spatially-Variant Deconvolution.

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

The software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the software. If not, see <http://www.gnu.org/licenses/>.
'''


from functools import reduce
from scipy import fftpack
from numpy.fft import rfft2,irfft2
import numpy as np
from scipy.interpolate import griddata
import logging
from data_utils import pickle_save, pickle_load, unpad
log = logging.getLogger('')

def compute_grid(psf_map, input_image):
    """
    Computes interpolation grid coefficients
    """
    grid_z1 = []
    grid_x, grid_y = np.mgrid[0:input_image.shape[0], 0:input_image.shape[1]]
    xmax = np.linspace(0, input_image.shape[0], psf_map.shape[0])
    ymax = np.linspace(0, input_image.shape[1], psf_map.shape[1])

    for i in range(psf_map.shape[0]*psf_map.shape[1]):
        log.info('Compute interpolation for patch:{}/{}'.format(i,psf_map.shape[0]*psf_map.shape[1]))
        points = []
        values = []
        for x in xmax:
            for y in ymax:
                points.append(np.asarray([x, y]))
                values.append(0.0)

        values[i] = 1.0

        points = np.asarray(points)
        values = np.asarray(values)

        grid_z1.append(griddata(points, values, (grid_x, grid_y), method='linear', rescale=True))
    pickle_save('grid_{}.pickle.gz'.format(psf_map.shape[0]*psf_map.shape[1]), grid_z1, compressed=True)

def load_grid(num_psf):
    """
    Load grid
    """
    log.info("Load Grid data")
    return pickle_load('grid_{}.pickle.gz'.format(num_psf), compressed=True)

def div0( a, b ):
    """
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def _normalize_kernel(kern):
    """
    Normalize kernels with sum is equal to one.
    """
    kern[kern < 0] = 0.0
    s = np.sum(kern, axis=(0, 1))
    kern = kern / s
    return kern


def _centered(arr, newshape):
    """
    Return the center newshape portion of the array.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    return reduce(np.add,np.gradient(F))

def rl_deconv_all(img_list, psf_list, iterations=10, lbd=0.2):
    """
    Spatially-Variant Richardson-lucy deconvolution with Total Variation regularization
    """
    min_value = []
    for img_idx, img in enumerate(img_list):
        img_list[img_idx] = np.pad(img_list[img_idx], np.max(psf_list[0].shape), mode='reflect')
        min_value.append(np.min(img))
        img_list[img_idx] = img_list[img_idx] - np.min(img)
    size = np.array(np.array(img_list[0].shape) + np.array(psf_list[0].shape)) - 1
    fsize = [fftpack.helper.next_fast_len(int(d)) for d in size]
    fslice = tuple([slice(0, int(sz)) for sz in size])

    latent_estimate = img_list.copy()
    error_estimate = img_list.copy()

    psf_f = []
    psf_flipped_f = []
    for img_idx, img in enumerate(latent_estimate):
        psf_f.append(rfft2(psf_list[img_idx], fsize))
        _psf_flipped = np.flip(psf_list[img_idx], axis=0)
        _psf_flipped = np.flip(_psf_flipped, axis=1)
        psf_flipped_f.append(rfft2(_psf_flipped, fsize))

    for i in range(iterations):
        log.info('RL TV Iter {}/{}, lbd = {}'.format(i, iterations, lbd))
        regularization = np.ones(img_list[0].shape)

        for img_idx, img in enumerate(latent_estimate):
            estimate_convolved = irfft2(np.multiply(psf_f[img_idx], rfft2(latent_estimate[img_idx], fsize)))[fslice].real
            estimate_convolved = _centered(estimate_convolved, img.shape)
            relative_blur = div0(img_list[img_idx], estimate_convolved)
            error_estimate[img_idx] = irfft2(np.multiply(psf_flipped_f[img_idx], rfft2(relative_blur, fsize)), fsize)[fslice].real
            error_estimate[img_idx] = _centered(error_estimate[img_idx], img.shape)
            regularization += 1.0 - (lbd * divergence(latent_estimate[img_idx] / np.linalg.norm(latent_estimate[img_idx], ord=1)))
            latent_estimate[img_idx] = np.multiply(latent_estimate[img_idx], error_estimate[img_idx])

        for img_idx, img in enumerate(img_list):
            latent_estimate[img_idx] = np.divide(latent_estimate[img_idx], regularization/float(len(img_list)))

    for img_idx, img in enumerate(latent_estimate):
        latent_estimate[img_idx] += min_value[img_idx]
        latent_estimate[img_idx] = unpad(latent_estimate[img_idx], np.max(psf_list[0].shape))

    return np.sum(latent_estimate, axis=0)