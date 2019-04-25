import os
from pathlib import Path

from PIL import Image

import torch
import torch.utils.data
from torchdataset import TorchDataSet
from torchvision import transforms


class Evaluator(object):

    def __init__(self, test, cuda):
        # define transform
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.loader = torch.utils.data.DataLoader(TorchDataSet(test['image'],
                                                                test[['number_digits', 'd1', 'd2', 'd3', 'd4']],
                                                                transform),
                                                   batch_size=128, shuffle=False)
        self.transform = transform
        # define device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device(cuda)

    def evaluate(self, model):
        corrects = 0
        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels) in enumerate(self.loader):
                # get batch data and move to device
                images = images.to(self.device)
                number_digits = length_labels.to(self.device)
                digits = [digit_labels.to(self.device) for digit_labels in digits_labels]
                # eval the model
                number_digits_pred, d1_pred, d2_pred, d3_pred, d4_pred = model.eval()(images)
                # get predictions
                number_digits = number_digits_pred.max(1)[1]
                d1 = d1_pred.max(1)[1]
                d2 = d2_pred.max(1)[1]
                d3 = d3_pred.max(1)[1]
                d4 = d4_pred.max(1)[1]
                # compare
                corrects += ((d1.eq(digits[0])) & (d2.eq(digits[1])) & (d3.eq(digits[2])) & (d4.eq(digits[3])))  .sum()
        return corrects.item() / len(self.loader.dataset)
