import argparse
import torch

from PIL import Image
from torchvision import transforms
import numpy
import cv2


class Detector(object):

    def __init__(self, classifier):
        self.classifier = classifier

    def detect(self, image, window_size=(100, 64), window_step=8, pyramid_scale=1.5, yield_first=False):
        detected_list = []
        for i, resized in enumerate(self._pyramid(image, scale=pyramid_scale, yield_first=yield_first)):
            #resized.show()
            #print(resized.size)
            for (x, y, window) in self._sliding_window(resized, step_size=window_step, window_size=window_size):
                if window.size[0] != window_size[0] or window.size[1] != window_size[1]:
                    continue
                nd, d1, d2, d3, d4, score = self.classifier.predict(window)
                #image = numpy.array(resized)# IDEA:
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
                #cv2.imshow('image', image)
                #cv2.waitKey(0)
                if nd >= 0 and (d1 != 0 or d2 != 0 or d3 != 0 or d4 != 0):
                    digits = [d1, d2, d3, d4]
                    digits = [0 if d == 10 else d for d in digits]
                    scale_power = i + 1 if yield_first == False else i
                    detected_list.append([nd, score, digits[0], digits[1], digits[2], digits[3], x, x + window_size[0], y, y + window_size[1], pyramid_scale, scale_power])
        detected = numpy.array(detected_list)
        max_nd = numpy.max(detected[:, 0])
        detected = detected[detected[:, 0] == max_nd]
        argsort = numpy.flip(numpy.argsort(detected[:, 1]))
        detected = detected[argsort]
        return detected[0]

    def _pyramid(self, image, scale=1.5, min_size=(64, 64), yield_first=False):
        if yield_first == True:
            yield image # return the first image in the generator
        while True:
            w, h = image.size
            w = int(w / scale)
            h = int(h / scale)
            image = image.resize([w, h])
            if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
                break
            yield image

    def _sliding_window(self, image, step_size, window_size):
        image = numpy.array(image)
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
    			# yield the current window
                yield (x, y, Image.fromarray(image[y:y + window_size[1], x:x + window_size[0]].astype(numpy.uint8)))
