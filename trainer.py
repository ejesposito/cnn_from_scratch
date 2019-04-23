import os
from pathlib import Path
import time

import h5py
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from torchdataset import TorchDataSet
from vgg16pretrained import VGG16Pretrained


class Trainer(object):

    def __init__(self, train, test):
        # load or create train dataset
        self.train = train
        self.test = test
        # define transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop([54, 54]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])
        validation_transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])
        # load the dataset and apply all the transformation in the GPU before creating the DataLoader
        train_torch_dataset = TorchDataSet(train['image'],
                                           train[['number_digits', 'd1', 'd2', 'd3', 'd4']],
                                           train_transform)
        test_torch_dataset = TorchDataSet(test['image'],
                                          test[['number_digits', 'd1', 'd2', 'd3', 'd4']],
                                          validation_transform)
        # create the data loaders
        self.train_loader = DataLoader(train_torch_dataset, batch_size=64, shuffle=True,
                                       num_workers=4)
        self.test_loader = DataLoader(test_torch_dataset, batch_size=64, shuffle=True,
                                    num_workers=4)
        # load the pretrained model
        self.vgg16_pretrained = VGG16Pretrained()
        for param in self.vgg16_pretrained.features.parameters():
            param.requires_grad = True
        print(self.vgg16_pretrained)
        #for param in vgg16_pretrained.features.parameters():
            #param.requires_grad = False
        # move model to GPU if CUDA is available
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')
        self.vgg16_pretrained.to(self.device)

    def train_nn(self):
        # select loss function
        criterion_transfer = nn.CrossEntropyLoss()
        # select optimizer
        optimizer_transfer = optim.SGD(self.vgg16_pretrained.classifier.parameters(), lr = 0.01)
        # optimizer_transfer = optim.Adam(self.vgg16_pretrained.classifier.parameters(), lr=0.0001, betas=(0.9, 0.999), amsgrad=True)
        # number of epochs
        n_epochs = 100
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
            number_correct_train = 0
            total_train = 0
            for batch_idx, (data, target) in enumerate(loaders):
                # move data to device
                target_nd, target_d1, target_d2, target_d3, target_d4 = target
                data = data.to(self.device)
                target_nd = target_nd.to(self.device)
                target_d1 = target_d1.to(self.device)
                target_d2 = target_d2.to(self.device)
                target_d3 = target_d3.to(self.device)
                target_d4 = target_d4.to(self.device)
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
                ## record the average training loss
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
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
                number_correct_train = number_correct_train + number_correct_this
                total_train = total_train + float(data.shape[0])
            ######################
            # validate the model #
            ######################
            model.eval()
            number_correct_test = 0
            total_test = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                target_nd, target_d1, target_d2, target_d3, target_d4 = target
                data = data.to(self.device)
                target_nd = target_nd.to(self.device)
                target_d1 = target_d1.to(self.device)
                target_d2 = target_d2.to(self.device)
                target_d3 = target_d3.to(self.device)
                target_d4 = target_d4.to(self.device)
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
                number_correct_test = number_correct_test + number_correct_this
                total_test = total_test + float(data.shape[0])
                #print('number correct this: {}'.format(number_correct_this))
            # print training statistics
            accuracy_train = number_correct_train.item() / total_train * 100.
            accuracy_test = number_correct_test.item() / total_test * 100.
            end = time.time()
            print('Train: correct {} - total {}'.format(number_correct_train, total_train))
            print('Test: correct {} - total {}'.format(number_correct_test, total_test))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTrainig Accuracy: {}% \t Validation Accuracy: {} \tTime: {}'.format(
                epoch,
                train_loss, valid_loss, accuracy_train, accuracy_test, (end - start) / 60
                ))
        # return trained model
        return model
