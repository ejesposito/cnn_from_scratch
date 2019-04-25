import os
import argparse
import time

import numpy
import cv2
from PIL import Image

import torch

from dataset import DataSet
from trainer import Trainer
from custommodel import CustomModel
from predictor import Predictor
from detector import Detector
from videohelper import image_helper, video_helper


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='detect')
parser.add_argument('--model', default='custom')
parser.add_argument('--cuda', default='cuda:0')
parser.add_argument('--n_workers', default=0, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--sgd_momentum', default=0.92, type=float)
parser.add_argument('--sgd_weight_decay', default=0.001, type=float)
parser.add_argument('--decay_steps', default=5000, type=int)
parser.add_argument('--decay_rate', default=0.94, type=float)


def main(args):
    training_options = {
        'mode': args.mode,
        'model': args.model,
        'cuda': args.cuda,
        'n_workers': args.n_workers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'momentum': args.sgd_momentum,
        'weight_decay': args.sgd_weight_decay,
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate,
    }

    if training_options['mode'] == 'train':
        # load the dataset in memory
        data_set = DataSet()
        # start train
        trainer = Trainer(data_set.train, data_set.test, training_options)
        trainer.fit()

    elif training_options['mode'] == 'detect':
        # load the model to compute predictions
        if training_options['model'] == 'custom':
            model = CustomModel()
            model.load_state_dict(torch.load(os.path.join('results', 'custom_model.pth'), map_location='cpu'))
        # load the images
        image_1 = Image.open(os.path.join('test_assets', '1.jpeg'))
        image_2 = Image.open(os.path.join('test_assets', '2.jpg'))
        image_3 = Image.open(os.path.join('test_assets', '3.jpg'))
        image_4 = Image.open(os.path.join('test_assets', '4.jpeg'))
        image_5 = Image.open(os.path.join('test_assets', '5.jpeg'))
        # detect the number for different images
        predictor = Predictor(model, training_options['cuda'])
        detector = Detector(predictor)
        print('Detecting numbers in 1.jpge (~3 min without GPU)')
        image_helper(detector, image_1, out_name='1')
        print('Detecting numbers in 2.jpge (~3 min without GPU)')
        image_helper(detector, image_2, out_name='2')
        print('Detecting numbers in 3.jpge (~3 min without GPU)')
        image_helper(detector, image_3, out_name='3')
        print('Detecting numbers in 4.jpge (~3 min without GPU)')
        image_helper(detector, image_4, window_size=(120, 80), out_name='4')
        print('Detecting numbers in 5.jpge (~3 min without GPU)')
        image_helper(detector, image_5, window_size=(65, 55), out_name='5')
        print('Done. Output in ./output/*')

    else: # video
        # load the model to compute predictions
        if training_options['model'] == 'custom':
            model = CustomModel()
            model.load_state_dict(torch.load(os.path.join('results', 'custom_model.pth'), map_location='cpu'))
        # generate the video
        predictor = Predictor(model, training_options['cuda'])
        detector = Detector(predictor)
        video_helper('video.mp4', 1, predictor, detector)


if __name__ == '__main__':
    main(parser.parse_args())
