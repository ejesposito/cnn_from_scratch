import os
import scipy.io as sc
import argparse

from PIL import Image
import cv2

import torch

from dataset import DataSet
from trainer import Trainer
from custommodel import CustomModel
from predictor import Predictor

parser = argparse.ArgumentParser()
parser.add_argument('-mode', '--mode', default='train')
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
parser.add_argument('-img', '--image')


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
        'image': args.image
    }

    if training_options['mode'] == 'train':
        # load the dataset in memory
        data_set = DataSet()
        # start train
        trainer = Trainer(data_set.train, data_set.test, training_options)
        trainer.fit()
    else:
        # load the model to compute predictions
        if training_options['model'] == 'custom':
            model = CustomModel()
            model.load_state_dict(torch.load(os.path.join('results', 'custom_model.pth'), map_location='cpu'))
        # load the images
        image = Image.open(os.path.join('test_assets', training_options['image']))
        #image.show()
        # compute the predictions
        predictor = Predictor(model, training_options['cuda'])
        predictor.predict(image)


if __name__ == '__main__':
    main(parser.parse_args())
