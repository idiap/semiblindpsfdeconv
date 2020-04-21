# Semi-blind Spatially-Variant Deconvolution
Code for the pytorch implementation of "Semi-Blind Spatially-Variant Deconvolution in Optical Microscopy with Local Point Spread Function Estimation By Use Of Convolutional Neural Networks"
https://ieeexplore.ieee.org/abstract/document/8451736
DOI: 10.1109/ICIP.2018.8451736 
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

## Citation
For any use of the code or parts of the code, please cite:
> @INPROCEEDINGS{8451736, author={A. {Shajkofci} and M. {Liebling}}, booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)}, title={Semi-Blind Spatially-Variant Deconvolution in Optical Microscopy with Local Point Spread Function Estimation by Use of Convolutional Neural Networks}, year={2018}, volume={}, number={}, pages={3818-3822},} 

## Licence
This is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. Parts of the code are coming from the PyTorch project (https://github.com/pytorch/).
