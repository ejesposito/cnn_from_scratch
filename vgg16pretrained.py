import os
from pathlib import Path

import h5py
import numpy as np
import cv2

class VGG16Pretrained(object):

    def __init__(self, train, test):
        # load or create train dataset
        self.train = train
        self.test = test
