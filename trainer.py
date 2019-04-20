import os
from pathlib import Path
import time

import h5py
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from torchdataset import TorchDataSet
from vgg16pretrained import VGG16Pretrained


class Trainer(object):

    def __init__(self, train, test):
        # load or create train dataset
        self.train = train
        self.test = test
        # create the data loader for torch
        #torch.multiprocessing.set_start_method("spawn")
        train_torch_dataset = TorchDataSet(train['image'],
                                           train[['number_digits', 'd1', 'd2', 'd3', 'd4']])
        test_torch_dataset = TorchDataSet(test['image'],
                                          test[['number_digits', 'd1', 'd2', 'd3', 'd4']])
        self.train_loader = DataLoader(train_torch_dataset, batch_size=64, shuffle=True,
                                       num_workers=0)
        self.test_loder = DataLoader(train_torch_dataset, batch_size=64, shuffle=True,
                                    num_workers=0)
        # load the pretrained model
        self.vgg16_pretrained = VGG16Pretrained()
        print(self.vgg16_pretrained)
        # move model to GPU if CUDA is available
        if torch.cuda.is_available():
            self.vgg16_pretrained.cuda(1)

    def train_nn(self):
        # select loss function
        criterion_transfer = nn.CrossEntropyLoss()
        # select optimizer
        optimizer_transfer = optim.SGD(self.vgg16_pretrained.classifier.parameters(), lr = 0.001)
        # number of epochs
        n_epochs = 100
        # train the model
        self.vgg16_pretrained = self._train(n_epochs, self.train_loader, self.vgg16_pretrained, optimizer_transfer, criterion_transfer, 'model_transfer.pt')

    def _train(self, n_epochs, loaders, model, optimizer, criterion, save_path):
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        for epoch in range(1, n_epochs+1):
            start = time.time()
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            ###################
            # train the model #
            ###################
            model.train()
            for batch_idx, (data, target) in enumerate(loaders):
                target_nd, target_d1, target_d2, target_d3, target_d4 = target
                ## find the loss and update the model parameters accordingly
                # reset the gradients
                optimizer.zero_grad()
                # obtain the prediction
                pred_nd, pred_d1, pred_d2, pred_d3, pred_d4 = model(data)
                # calculate the error
                loss_nd = criterion(pred_nd, target_nd)
                loss_d1 = criterion(pred_d1, target_d1)
                loss_d2 = criterion(pred_d2, target_d2)
                loss_d3 = criterion(pred_d3, target_d3)
                loss_d4 = criterion(pred_d4, target_d4)
                loss = loss_nd + loss_d1 + loss_d2 + loss_d3 + loss_d4
                # backpropagate the error
                loss.backward()
                optimizer.step()
                ## record the average training loss, using something like
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            # print training statistics
            end = time.time()
            print('Epoch: {} \tTraining Loss: {:.6f} \tTime: {}'.format(
                epoch,
                train_loss, (end - start) / 60
                ))
        # return trained model
        return model
