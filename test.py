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

import sys
from data_utils import load
import logging
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch import FloatTensor
import numpy as np
from skimage.transform import downscale_local_mean
import scipy
from deconvolution import load_grid, compute_grid, rl_deconv_all
from data_utils import scale, gaussian_kernel
from skimage import io


if (len(sys.argv) > 1):
    isGrid = 1
    run_nb = str(sys.argv[2])
else:
    isGrid = 0
    run_nb = 'local'

logging.basicConfig(
    format="%(asctime)s [RUN {}] %(message)s".format(run_nb),
    handlers=[
        logging.FileHandler("output_log_{}.log".format(run_nb)),
        logging.StreamHandler()
    ])

log = logging.getLogger('')
log.setLevel(logging.INFO)
log.info("Starting test {}....".format(run_nb))


def live_moving_window(im, step=64):
    """
    Computes the focus paramter map on the input image using a moving window
    :param im: input impate
    :param step: resolution of the moving average window
    :return:
    """
    size = 128
    num_classes = 2
    x = size
    y = size
    im = im[0:im.shape[0]//size * size, 0:im.shape[1]//size * size]
    weight_image = np.zeros((im.shape[0], im.shape[1], num_classes))

    tile_dataset = []
    y_size = 0
    i = 0

    while x <= im.shape[0]:
        x_size = 0
        while y <= im.shape[1]:
            a = im[x - size:x, y - size:y]
            a = scale(a)
            tile_dataset.append(a[:])
            weight_image[x - size:x, y - size:y] += 1.0
            y += step
            x_size += 1
            i += 1
        y = size
        y_size += 1
        x += step

    tile_dataset = np.asarray(tile_dataset)
    tile_dataset = np.reshape(tile_dataset, (tile_dataset.shape[0], 1, size, size))
    input_tensor = FloatTensor(tile_dataset)
    out = model(Variable(input_tensor).cuda())
    output_npy = out.data.cpu().numpy()
    output = np.zeros((im.shape[0], im.shape[1], output_npy.shape[1]))

    i = 0
    x = size
    y = size
    while x <= im.shape[0]:
        while y <= im.shape[1]:
            output[x - size:x, y - size:y] += output_npy[i, :]
            y += step
            i += 1
        y = size
        x += step

    output = output / weight_image
    return output, output_npy


def deconvolution_demo(first_img):
    """
    Computes the spatially-variant focus parameters and deconvolve the image
    :param first_img: input image
    """
    model.train(False)
    step = 64
    psf_array_linear = []
    image_masked_array_bilinear=[]

    # Get focus map
    downscaled = first_img[128:-128, 128:-128]
    downscaled = scale(downscale_local_mean(downscaled, (3, 3)))

    img, _ = live_moving_window(downscaled, step)
    img_downsampled = img[::step, ::step]
    output_filtered = [scipy.ndimage.filters.median_filter(img_downsampled[:,:,i], size=(2,2), mode='reflect') for i in range(img.shape[2])]
    output_filtered_scaled = [downscale_local_mean(output_filtered[i], (2,2)) for i in range(len(output_filtered))]
    flattened_map = [output_filtered_scaled[i].flatten() for i in range(len(output_filtered_scaled))]

    # Synthesize kernels
    for i in range(0, flattened_map[0].shape[0]):
        new_psf = gaussian_kernel(63, flattened_map[0][i], flattened_map[1][i])
        psf_array_linear.append(new_psf)
    psf_array_linear = np.asarray(psf_array_linear)

    # Deconvolve
    compute_grid(output_filtered_scaled[0],downscaled)
    grid_z1 = load_grid(psf_array_linear.shape[0])
    for i, current_psf in enumerate(psf_array_linear):
        log.info('Detected PSF {} with focus x {} y {}'.format(i, flattened_map[0][i],flattened_map[1][i]))
        image_masked_array_bilinear.append(np.multiply(grid_z1[i], downscaled))

    deconvolved = rl_deconv_all(image_masked_array_bilinear, psf_array_linear, iterations=20, lbd=0.1)

    # Figures
    io.imsave('original.png', scale(downscaled))
    io.imsave('deconvolved.png', scale(deconvolved))
    plt.figure()
    fig, ax = plt.subplots()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 2)
    ax4 = plt.subplot(2, 2, 4)

    im1 = ax1.imshow(downscaled)
    ax1.set_title('Original')
    im2 = ax2.imshow(output_filtered_scaled[0][:,:], vmin=1, vmax=6)
    ax2.set_title('Detected FWMH X (px)')
    im3 = ax3.imshow(deconvolved)
    ax3.set_title('Deconvolved RL TV')
    im4 = ax4.imshow(output_filtered_scaled[1][:,:], vmin=1, vmax=6)
    ax4.set_title('Detected FWMH Y (px)')
    plt.show()

if __name__ == "__main__":
    model = load('models/model_26.pt')
    image = io.imread('data/fly.png')
    deconvolution_demo(image)