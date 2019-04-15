import os
import scipy.io as sc

import cv2

from dataset import DataSet
from vgg16pretrained import VGG16Pretrained


if __name__ == '__main__':
    # load the dataset in memory
    data_set = DataSet()
    # print dataset stats
    print('Train size: {}'.format(data_set.train.shape))
    print('Test size: {}'.format(data_set.test.shape))
    # example of use of the nympy structured array for train
    image = data_set.train[0]['image']
    print('Train image number of digits: {}'.format(data_set.train[0]['number_digits']))
    print('D1: {}'.format(data_set.train[0]['d1']))
    print('D2: {}'.format(data_set.train[0]['d2']))
    print('D3: {}'.format(data_set.train[0]['d3']))
    print('D4: {}'.format(data_set.train[0]['d4']))
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    # example of use of the nympy structured array for test
    image = data_set.test[0]['image']
    print('Test image number of digits: {}'.format(data_set.test[0]['number_digits']))
    print('D1: {}'.format(data_set.test[0]['d1']))
    print('D2: {}'.format(data_set.test[0]['d2']))
    print('D3: {}'.format(data_set.test[0]['d3']))
    print('D4: {}'.format(data_set.test[0]['d4']))
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    # train the classifier
    vgg16_pretrained = VGG16Pretrained(data_set.train, data_set.test)
