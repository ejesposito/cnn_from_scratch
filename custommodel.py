import os

import torch
import torch.nn as nn

class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.seq_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.seq_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.seq_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.seq_5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.4)
        )
        self.seq_6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.4)
        )
        self.seq_7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.4)
        )
        self.seq_8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.4)
        )
        self.seq_9 = nn.Sequential(
            nn.Linear(9408, 3072),
            nn.ReLU()
        )
        self.seq_10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )
        self.seq_n_digits = nn.Sequential(nn.Linear(3072, 5))
        self.seq_d1 = nn.Sequential(nn.Linear(3072, 11))
        self.seq_d2 = nn.Sequential(nn.Linear(3072, 11))
        self.seq_d3 = nn.Sequential(nn.Linear(3072, 11))
        self.seq_d4 = nn.Sequential(nn.Linear(3072, 11))

    def forward(self, x):
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)
        x = self.seq_4(x)
        x = self.seq_5(x)
        x = self.seq_6(x)
        x = self.seq_7(x)
        x = self.seq_8(x)
        x = x.view(x.size(0), -1)
        x = self.seq_9(x)
        out = self.seq_10(x)
        number_digits = self.seq_n_digits(out)
        d1 = self.seq_d1(out)
        d2 = self.seq_d2(out)
        d3 = self.seq_d3(out)
        d4 = self.seq_d4(out)
        return number_digits, d1, d2, d3, d4
