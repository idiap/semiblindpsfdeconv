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
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging
from data_utils import load_crops, save
from model import resnet50

if (len(sys.argv) > 1):
    isGrid = 1
    run_nb = str(sys.argv[2])
else:
    isGrid = 0
    run_nb = 'local'

patch_size = None

logging.basicConfig(
    format="%(asctime)s [RUN {}] %(message)s".format(run_nb),
    handlers=[
        logging.FileHandler("output_log_{}.log".format(run_nb)),
        logging.StreamHandler()
    ])

log = logging.getLogger('')
log.setLevel(logging.INFO)
log.info("Starting run {}....".format(run_nb))


def train():

    l2loss = nn.MSELoss(size_average=True).cuda()
    model.train(True)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)

    def adjust_learning_rate(epoch):
        lr = learning_rate * (0.9 ** (epoch // 2))
        return lr

    for ep in range(epoch):
        batch_nb = 0
        cum_loss = 0
        lr = adjust_learning_rate(ep)
        log.info("Learning rate : {}".format(lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        image_num = 0
        for image, label in train_loader:
            image = Variable(image).cuda()
            optimizer.zero_grad()
            output = model(image)
            loss = l2loss(output, Variable(label.float()).cuda())
            loss.backward()
            optimizer.step()

            cum_loss += loss.cpu().data.numpy()
            batch_nb += 1
            image_num += output.size(0)

            log.info(
                "Ep {0}/{1}, lr {6}, bt {2}/{3}, loss {4}, avg loss {5}".format(ep, epoch, batch_nb, len(train_loader),
                                                                                loss.cpu().data.numpy(),
                                                                                cum_loss / batch_nb, lr))

        save(model, run_nb)
    log.info("The training is complete.")



if __name__== "__main__":
    epoch = 10
    batch_size = 8
    learning_rate = 0.001

    train_loader, test_loader, train_file, test_file = load_crops(128)
    model = resnet50().cuda()
    train()
