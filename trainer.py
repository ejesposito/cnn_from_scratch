import argparse
import os
import time
from datetime import datetime

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional
from torchdataset import TorchDataSet
from torchvision import transforms
import torch.utils.data

from custommodel import CustomModel
from vgg16pretrained import VGG16Pretrained
from evaluator import Evaluator


class Trainer(object):

    def __init__(self, train, test, params):
        self.train = train
        self.test = test
        self.params = params

    def fit(self):
        # Get training parameters
        model_name = self.params['model']
        cuda = self.params['cuda']
        n_workers = self.params['n_workers']
        n_epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        initial_learning_rate = self.params['learning_rate']
        momentum = self.params['momentum']
        weight_decay = self.params['weight_decay']
        decay_steps = self.params['decay_steps']
        decay_rate = self.params['decay_rate']
        # Set the device
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device(cuda)
        # Set the model archotecture to train
        if model_name == 'custom':
            model = CustomModel()
        elif model_name == 'vgg16_pretrained':
            model = VGG16Pretrained(use_pretrained_weights=True)
        elif model_name == 'vgg16':
            model = VGG16Pretrained(use_pretrained_weights=False)
        else:
            model = CustomModel()
        print('Selected model: {}'.format(model_name))
        model.to(device)
        # Create the transform
        transform = transforms.Compose([
            transforms.RandomCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_loader = torch.utils.data.DataLoader(TorchDataSet(self.train['image'],
                                                                self.train[['number_digits', 'd1', 'd2', 'd3', 'd4']],
                                                                transform),
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=n_workers, pin_memory=True)
        # Define the optimizer
        optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
        evaluator = Evaluator(self.test, cuda)
        # Run the epochs
        model_accuracy = 0
        losses = np.empty([0], dtype=np.float32)
        for i in range(n_epochs):
            # Execute one epoch
            print('Epoch number: {}'.format(i))
            for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
                # Execute one batch
                images = images.to(device)
                number_digits = length_labels.to(device)
                digits = [digit_labels.to(device) for digit_labels in digits_labels]
                # Get predictions
                pred_number_digits, pred_d1, pred_d2, pred_d3, pred_d4 = model.train()(images)
                # Learn
                loss = self._compute_loss(pred_number_digits, pred_d1, pred_d2, pred_d3, pred_d4, number_digits, digits)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            # print epoch's results
            losses = np.append(losses, loss.item())
            np.save(os.path.join('results', model_name + '_losses.npy'), losses)
            accuracy = evaluator.evaluate(model)
            print('Test loss: {}  |  Test accuracy: {}'.format(loss.item(), accuracy))
            if accuracy > model_accuracy:
                print('Better accuracy found. Model saved.')
                torch.save(model.state_dict(), os.path.join('results', model_name + '.pth'))
                best_accuracy = accuracy

    def _compute_loss(self, pred_number_digits, pred_d1, pred_d2, pred_d3, pred_d4, number_digits, digits):
        length_cross_entropy = torch.nn.functional.cross_entropy(pred_number_digits, number_digits)
        digit1_cross_entropy = torch.nn.functional.cross_entropy(pred_d1, digits[0])
        digit2_cross_entropy = torch.nn.functional.cross_entropy(pred_d2, digits[1])
        digit3_cross_entropy = torch.nn.functional.cross_entropy(pred_d3, digits[2])
        digit4_cross_entropy = torch.nn.functional.cross_entropy(pred_d4, digits[3])
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy
        return loss
