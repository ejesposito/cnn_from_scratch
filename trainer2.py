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

from dataset2 import TorchDataSet
from evaluator2 import Evaluator
from model2 import Model


def _loss(length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, length_labels, digits_labels):
    length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, length_labels)
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, digits_labels[1])
    digit3_cross_entropy = torch.nn.functional.cross_entropy(digit3_logits, digits_labels[2])
    digit4_cross_entropy = torch.nn.functional.cross_entropy(digit4_logits, digits_labels[3])
    loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy
    return loss


def _train(training_options, train, test):
    batch_size = training_options['batch_size']
    initial_learning_rate = training_options['learning_rate']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    step = 0
    patience = initial_patience
    best_accuracy = 0.0
    duration = 0.0

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:1')

    model = Model()
    model.to(device)

    transform = transforms.Compose([
        transforms.RandomCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_loader = torch.utils.data.DataLoader(TorchDataSet(train['image'],
                                                            train[['number_digits', 'd1', 'd2', 'd3', 'd4']],
                                                            transform),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)
    evaluator = Evaluator(test)
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=training_options['decay_steps'], gamma=training_options['decay_rate'])

    losses = np.empty([0], dtype=np.float32)

    while True:
        for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
            start_time = time.time()
            images, length_labels, digits_labels = images.to(device), length_labels.to(device), [digit_labels.to(device) for digit_labels in digits_labels]
            length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits = model.train()(images)
            loss = _loss(length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, length_labels, digits_labels)

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

            if step % num_steps_to_check != 0:
                continue

            losses = np.append(losses, loss.item())

            print('=> Evaluating on validation dataset...')
            accuracy = evaluator.evaluate(model)
            print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

            if accuracy > best_accuracy:
                print('=> Model saved to file')
                patience = initial_patience
                best_accuracy = accuracy
            else:
                patience -= 1

            print('=> patience = %d' % patience)
            if patience == 0:
                return
