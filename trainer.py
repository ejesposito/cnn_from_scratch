import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from torchdataset import TorchDataSet
from custommodel import CustomModel
from vgg16pretrained import VGG16Pretrained
from evaluator import Evaluator


class Trainer(object):

    def __init__(self, train, test, params):
        self.train = train
        self.test = test
        self.params = params

    def fit(self):
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

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device(cuda)

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
        evaluator = Evaluator(self.test, cuda)
        optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
        losses = np.empty([0], dtype=np.float32)

        num_steps_to_show_loss = 100
        num_steps_to_check = 1000
        step = 0
        best_accuracy = 0.0
        duration = 0.0

        for i in range(n_epochs):
            print('=> Starting epoch: {}'.format(i))
            for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
                start_time = time.time()
                images = images.to(device)
                length_labels = length_labels.to(device)
                digits_labels = [digit_labels.to(device) for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits = model.train()(images)
                loss = self._compute_loss(length_logits, digit1_logits, digit2_logits, digit3_logits,
                                          digit4_logits, length_labels, digits_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                duration += time.time() - start_time

                if step % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print('=> %s: step %d, loss = %f, learning_rate = %f (%.1f examples/sec)' % (
                        datetime.now(), step, loss.item(), scheduler.get_lr()[0], examples_per_sec))

                if step % num_steps_to_check == 0:
                    losses = np.append(losses, loss.item())
                    np.save(os.path.join('results', model_name + '_losses.npy'), losses)
                    print('=> Evaluating on validation dataset...')
                    accuracy = evaluator.evaluate(model)
                    print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))
                    if accuracy > best_accuracy:
                        print('=> Model saved to file')
                        torch.save(model.state_dict(), os.path.join('results', model_name + '.pth'))
                        best_accuracy = accuracy

    def _compute_loss(self, length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, length_labels, digits_labels):
        length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, length_labels)
        digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, digits_labels[0])
        digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, digits_labels[1])
        digit3_cross_entropy = torch.nn.functional.cross_entropy(digit3_logits, digits_labels[2])
        digit4_cross_entropy = torch.nn.functional.cross_entropy(digit4_logits, digits_labels[3])
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy
        return loss
