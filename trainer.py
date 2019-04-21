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
        self.test_loader = DataLoader(test_torch_dataset, batch_size=64, shuffle=True,
                                    num_workers=0)
        # load the pretrained model
        self.vgg16_pretrained = VGG16Pretrained()
        print(self.vgg16_pretrained)
        #for param in vgg16_pretrained.features.parameters():
            #param.requires_grad = False
        # move model to GPU if CUDA is available
        if torch.cuda.is_available():
            self.vgg16_pretrained.cuda(1)

    def train_nn(self):
        # select loss function
        criterion_transfer = nn.CrossEntropyLoss()
        # select optimizer
        optimizer_transfer = optim.SGD(self.vgg16_pretrained.classifier.parameters(), lr = 0.1)
        # number of epochs
        n_epochs = 200
        # train the model
        self.vgg16_pretrained = self._train(n_epochs, self.train_loader, self.test_loader, self.vgg16_pretrained, optimizer_transfer, criterion_transfer, 'model_transfer.pt')

    def _train(self, n_epochs, loaders, test_loader, model, optimizer, criterion, save_path):
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
            ######################
            # validate the model #
            ######################
            model.eval()
            number_correct = 0
            total = 0
            print('Number correct before validate: {}'.format(number_correct))
            for batch_idx, (data, target) in enumerate(test_loader):
                target_nd, target_d1, target_d2, target_d3, target_d4 = target
                ## update the average validation loss
                # obtain the prediction
                pred_nd, pred_d1, pred_d2, pred_d3, pred_d4 = model(data)
                # calculate the error
                loss_nd = criterion(pred_nd, target_nd)
                loss_d1 = criterion(pred_d1, target_d1)
                loss_d2 = criterion(pred_d2, target_d2)
                loss_d3 = criterion(pred_d3, target_d3)
                loss_d4 = criterion(pred_d4, target_d4)
                loss = loss_nd + loss_d1 + loss_d2 + loss_d3 + loss_d4
                # record the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                # eval
                length_prediction = pred_nd.max(1)[1]
                digit1_prediction = pred_d1.max(1)[1]
                digit2_prediction = pred_d2.max(1)[1]
                digit3_prediction = pred_d3.max(1)[1]
                digit4_prediction = pred_d4.max(1)[1]
                #print('pred nd: {}'.format(pred_nd))
                #print('max nd pred tensor: {}'.format(length_prediction))
                #print('nd target tensor: {}'.format(target_nd))
                number_correct_this = (length_prediction.eq(target_nd) &
                                       digit1_prediction.eq(target_d1) &
                                       digit2_prediction.eq(target_d2) &
                                       digit3_prediction.eq(target_d3) &
                                       digit4_prediction.eq(target_d4)).sum()
                number_correct = number_correct + number_correct_this
                total = total + float(data.shape[0])
                #print('number correct this: {}'.format(number_correct_this))
            # print training statistics
            accuracy = number_correct.item() / total * 100.
            end = time.time()
            print(total)
            print(number_correct)
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {}%\tTime: {}'.format(
                epoch,
                train_loss, valid_loss, accuracy, (end - start) / 60
                ))
        # return trained model
        return model
