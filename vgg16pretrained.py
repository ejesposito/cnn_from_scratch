import os
from pathlib import Path

import h5py
import numpy as np
import cv2

from torchdataset import TorchDataSet
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models


class VGG16Pretrained(nn.Module):

    def __init__(self):
        super(VGG16Pretrained, self).__init__()
        """
        model = models.vgg16(pretrained=True)
        first_layer_output = model.classifier[0].out_features
        last_layer_input = model.classifier[-1].in_features

        first_layer = nn.Linear(512, first_layer_output)
        classifier[0] = first_layer

        layer_number_digits = nn.Linear(last_layer_input, 5)
        layer_d1 = nn.Linear(last_layer_input, 11)
        layer_d2 = nn.Linear(last_layer_input, 11)
        layer_d3 = nn.Linear(last_layer_input, 11)
        layer_d4 = nn.Linear(last_layer_input, 11)
        """

        vgg16 = models.vgg16_bn(pretrained=True)
        print(vgg16)
        # copy vgg16 features and create sequence
        self.features = nn.Sequential(*list(vgg16.features.children()))
        # copy vgg16 classifier except the last layer
        classifier = list(vgg16.classifier.children())[:-1]
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        # modify first layer and create sequence
        first_layer = nn.Linear(512*7*7, 4096, bias=True)
        second_layer = nn.BatchNorm1d(4096)
        third_layer = nn.ReLU()
        fourth_layer = nn.Linear(4096, 4096, bias=True)
        fifth_layer = nn.BatchNorm1d(4096)
        sixth_layer = nn.ReLU()
        self.classifier = nn.Sequential(*[first_layer, second_layer, third_layer, fourth_layer, fifth_layer, sixth_layer])
        # create last layers
        self.layer_number_digits = nn.Linear(4096, 5)
        self.layer_d1 = nn.Linear(4096, 11)
        self.layer_d2 = nn.Linear(4096, 11)
        self.layer_d3 = nn.Linear(4096, 11)
        self.layer_d4 = nn.Linear(4096, 11)

    def forward(self, x):
        out1 = self.features(x)
        out1 = self.avgpool(out1)
        out1 = out1.view(x.size(0), -1)
        out2 = self.classifier(out1)
        number_digits = self.layer_number_digits(out2)
        d1 = self.layer_d1(out2)
        d2 = self.layer_d2(out2)
        d3 = self.layer_d3(out2)
        d4 = self.layer_d4(out2)
        return number_digits, d1, d2, d3, d4
