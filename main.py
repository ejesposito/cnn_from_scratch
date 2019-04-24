import os
import scipy.io as sc
import argparse

import cv2

from dataset import DataSet
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='custom')
parser.add_argument('-c', '--cuda', default='cuda:1')
parser.add_argument('-w', '--n_workers', default=0, type=int)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-bs', '--batch_size', default=32, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
parser.add_argument('-sm', '--sgd_momentum', default=0.9, type=float)
parser.add_argument('-sw', '--sgd_weight_decay', default=0.0005, type=float)
parser.add_argument('-ds', '--decay_steps', default=10000, type=int)
parser.add_argument('-dr', '--decay_rate', default=0.9, type=float)


def main(args):
    # args
    training_options = {
        'model': args.model,
        'cuda': args.cuda,
        'n_workers': args.n_workers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'momentum': args.sgd_momentum,
        'weight_decay': args.sgd_weight_decay,
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate
    }
    # load the dataset in memory
    data_set = DataSet()
    # start train
    trainer = Trainer(data_set.train, data_set.test, training_options)
    trainer.fit()


if __name__ == '__main__':
    main(parser.parse_args())
