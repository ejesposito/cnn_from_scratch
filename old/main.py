import os
import scipy.io as sc

import cv2

from dataset import DataSet
from trainer import Trainer

if __name__ == '__main__':
    # load the dataset in memory
    data_set = DataSet()
    # print dataset stats
    print('Train size: {}'.format(data_set.train.shape))
    print('Test size: {}'.format(data_set.test.shape))
    # print samples of the train data set
    #print('Train samples')
    #data_set.print_samples(data_set.train, num_samples=20)
    # print samples of the test data set
    #print('Test samples')
    #data_set.print_samples(data_set.test, num_samples=20)
    # train the classifier
    trainer = Trainer(data_set.train, data_set.test)
    trainer.train_nn()
