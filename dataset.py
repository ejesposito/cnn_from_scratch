import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import cv2


class DataSet(object):

    TRAIN_DATASET_DIR = 'dataset/train'
    TEST_DATASET_DIR = 'dataset/test'
    PREPROCESSED_DIR = 'preprocessed'
    TRAIN_NPY = 'preprocessed/train.npy'
    TEST_NPY = 'preprocessed/test.npy'

    def __init__(self):
        # load or create train dataset
        train_file = Path(self.TRAIN_NPY)
        if not train_file.is_file():
            print('Creating train.npy')
            self.train = self._create_preprocessed_dataset(self.TRAIN_DATASET_DIR,
                                                           self.PREPROCESSED_DIR, file_name='train.npy')
        else:
            print('Loading train.npy')
            self.train = np.load(self.TRAIN_NPY)
        # load or create test dataset
        test_file = Path(self.TEST_NPY)
        if not test_file.is_file():
            print('Creating test.npy')
            self.test = self._create_preprocessed_dataset(self.TEST_DATASET_DIR,
                                                          self.PREPROCESSED_DIR, file_name='test.npy')
        else:
            print('Loading test.npy')
            self.test = np.load(self.TEST_NPY)

    def _get_box_data(self, index, hdf5_data):
        """
        get `left, top, width, height` of each picture
        :param index:
        :param hdf5_data:
        :return:
        """
        meta_data = dict()
        meta_data['height'] = []
        meta_data['label'] = []
        meta_data['left'] = []
        meta_data['top'] = []
        meta_data['width'] = []

        def print_attrs(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(obj[0][0])
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(hdf5_data[obj[k][0]][0][0]))
            meta_data[name] = vals

        box = hdf5_data['/digitStruct/bbox'][index]
        hdf5_data[box[0]].visititems(print_attrs)
        return meta_data

    def _get_name(self, index, hdf5_data):
        name = hdf5_data['/digitStruct/name']
        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

    def _create_preprocessed_dataset(self, dataset_dir, preprocessed_dir, file_name):
        # create numpy structure dtype
        image_dt = np.dtype((np.float32, (48,48,3)))
        structure = [('image', image_dt), ('number_digits', np.int32),
                     ('d1', np.int32), ('d2', np.int32),
                     ('d3', np.int32), ('d4', np.int32)]
        # laod data into numpy struct
        mat_data = h5py.File(os.path.join(dataset_dir, 'digitStruct.mat'))
        size = mat_data['/digitStruct/name'].size
        dataset = np.empty(size, dtype=structure)
        for i in range(size):
            img_name = self._get_name(i, mat_data)
            img_data = self._get_box_data(i, mat_data)
            raw_img = cv2.imread(os.path.join(dataset_dir, img_name))
            left = max([0, np.min(np.array(img_data['left'], dtype=np.int16))])
            top = max([0, np.min(np.array(img_data['top'], dtype=np.int16))])
            right = np.max(np.array(img_data['left'], dtype=np.int16) + \
                           np.array(img_data['width'], dtype=np.int16))
            bottom = np.max(np.array(img_data['top'], dtype=np.int16) + \
                            np.array(img_data['height'], dtype=np.int16))
            cropped_img = raw_img[top:bottom, left:right]
            resized_img = cv2.resize(cropped_img, (48, 48))
            rescaled_image = cv2.normalize(resized_img, None, alpha=0, beta=1,
                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # transform image from BGR (opencv) to RGB (pil, used by pytorch)
            #rescaled_image_blue = rescaled_image[:,:,0]
            #rescaled_image[:,:,0] = rescaled_image[:,:,2]
            #rescaled_image[:,:,2] = rescaled_image_blue
            rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB)
            # normalize image with mean and std
            normalized_image = rescaled_image - np.array([0.485, 0.456, 0.406])
            normalized_image = normalized_image / np.array([0.229, 0.224, 0.225])
            #cv2.imshow('image', resized_img)
            #cv2.waitKey(0)
            dataset[i]['image'] = normalized_image
            dataset[i]['number_digits'] = len(img_data['label'])
            if dataset[i]['number_digits'] > 4:
                dataset[i]['d1'] = 10
                dataset[i]['d2'] = 10
                dataset[i]['d3'] = 10
                dataset[i]['d4'] = 10
            elif dataset[i]['number_digits'] == 4:
                dataset[i]['d1'] = img_data['label'][0]
                dataset[i]['d2'] = img_data['label'][1]
                dataset[i]['d3'] = img_data['label'][2]
                dataset[i]['d4'] = img_data['label'][3]
            elif dataset[i]['number_digits'] == 3:
                dataset[i]['d1'] = 10
                dataset[i]['d2'] = img_data['label'][0]
                dataset[i]['d3'] = img_data['label'][1]
                dataset[i]['d4'] = img_data['label'][2]
            elif dataset[i]['number_digits'] == 2:
                dataset[i]['d1'] = 10
                dataset[i]['d2'] = 10
                dataset[i]['d3'] = img_data['label'][0]
                dataset[i]['d4'] = img_data['label'][1]
            elif dataset[i]['number_digits'] == 1:
                dataset[i]['d1'] = 10
                dataset[i]['d2'] = 10
                dataset[i]['d3'] = 10
                dataset[i]['d4'] = img_data['label'][0]
        dataset = dataset[dataset['number_digits'] <= 4]
        np.save(os.path.join(preprocessed_dir, file_name), dataset)
        return dataset
