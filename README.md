# Semi-blind Spatially-Variant Deconvolution
Code for the pytorch implementation of "Semi-Blind Spatially-Variant Deconvolution in Optical Microscopy with Local Point Spread Function Estimation By Use Of Convolutional Neural Networks"

https://arxiv.org/abs/1803.07452

## Abstract
We present a semi-blind, spatially-variant deconvolution technique aimed at optical microscopy that combines a local estimation step of the point spread function (PSF) and deconvolution using a spatially variant, regularized Richardson-Lucy algorithm. To find the local PSF map in a computationally tractable way, we train a convolutional neural network to perform regression of an optical parametric model on synthetically blurred image patches. We deconvolved both synthetic and experimentally-acquired data, and achieved an improvement of image SNR of 1.00 dB on average, compared to other deconvolution algorithms.

## Adaptations
The physical model for the PSF here is an anisotropic Gaussian distribution, implemented in `gaussian_kernel`. It can be adapted easily to other models. Software releases with other types of models will be available in the following months.

## Requirements
The following python libraries are required. We advise the use of the conda package manager.
> numpy
> scipy
> scikit-image
> pandas
> pytorch
> matplotlib

## Howto
`train.py` is the file for training. `test.py` is the file for testing and deconvolution.

## Licence
This is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. Parts of the code are coming from the PyTorch project (https://github.com/pytorch/).
