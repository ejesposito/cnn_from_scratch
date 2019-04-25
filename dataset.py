import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import cv2


class DataSet(object):

    TRAIN_DATASET_DIR = 'dataset/train'
    TEST_DATASET_DIR = 'dataset/test'
    EXTRA_DATASET_DIR = 'dataset/extra'
    PREPROCESSED_DIR = 'preprocessed'
    TRAIN_NPY = 'preprocessed/train.npy'
    TEST_NPY = 'preprocessed/test.npy'

    def __init__(self):
        # load or create train dataset
        train_file = Path(self.TRAIN_NPY)
        if not train_file.is_file():
            print('Creating train.npy')
            self.train = self._create_preprocessed_dataset(self.TRAIN_DATASET_DIR,
                                                           add_negative_samples = True)
            self.extra = self._create_preprocessed_dataset(self.EXTRA_DATASET_DIR,
                                                           add_negative_samples = False)
            self.train = self._stack_structured_arrays(self.train, self.extra)
            np.save(os.path.join(self.PREPROCESSED_DIR, 'train.npy'), self.train)
        else:
            print('Loading train.npy')
            self.train = np.load(self.TRAIN_NPY)
        # load or create test dataset
        test_file = Path(self.TEST_NPY)
        if not test_file.is_file():
            print('Creating test.npy')
            self.test = self._create_preprocessed_dataset(self.TEST_DATASET_DIR,
                                                          add_negative_samples = False)
            np.save(os.path.join(self.PREPROCESSED_DIR, 'test.npy'), self.test)
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

    def _create_preprocessed_dataset(self, dataset_dir, add_negative_samples = False):
        # create numpy structure dtype
        image_dt = np.dtype((np.float32, (64,64,3)))
        structure = [('image', image_dt), ('number_digits', np.int32),
                     ('d1', np.int32), ('d2', np.int32),
                     ('d3', np.int32), ('d4', np.int32)]
        # laod data into numpy struct
        mat_data = h5py.File(os.path.join(dataset_dir, 'digitStruct.mat'))
        size = mat_data['/digitStruct/name'].size
        dataset = np.empty(size, dtype=structure)
        dataset['number_digits'] = -1
        dataset_negative_samples = np.empty(size, dtype=structure)
        number_negative_samples = 0
        for i in range(size):
            img_name = self._get_name(i, mat_data)
            img_data = self._get_box_data(i, mat_data)
            # get the image
            raw_image = Image.open(os.path.join(dataset_dir, img_name)) # PIL
            image_width, image_height = raw_image.size
            # get min bounding box that contains the digits
            left = max([0, np.min(np.array(img_data['left'], dtype=np.int16))])
            top = max([0, np.min(np.array(img_data['top'], dtype=np.int16))])
            right = np.max(np.array(img_data['left'], dtype=np.int16) + \
                           np.array(img_data['width'], dtype=np.int16))
            right = min([right, image_width])
            bottom = np.max(np.array(img_data['top'], dtype=np.int16) + \
                            np.array(img_data['height'], dtype=np.int16))
            bottom = min([bottom, image_height])
            box_width = right - left
            box_height = bottom - top
            # expand the bounding box 30% in x and y
            width_to_expand = int(box_width * 0.3 / 2)
            height_to_expand = int(box_height * 0.3 / 2)
            left = max(0, left - width_to_expand)
            right = min(right + width_to_expand, image_width)
            top = max(0, top - height_to_expand)
            bottom = min(bottom + height_to_expand, image_height)
            # image preprocessing
            img = raw_image.crop([left, top, right, bottom])
            img = img.resize([64, 64])
            np_img = np.array(img)
            # labeling
            dataset[i]['image'] = np_img.astype(np.float32)
            #img.show()
            #cv2.imshow('image2', dataset[i]['image'].astype(np.uint8))
            #cv2.waitKey(0)
            dataset[i]['number_digits'] = len(img_data['label'])
            #print(img_data['label'])
            if dataset[i]['number_digits'] > 4:
                dataset[i]['number_digits'] = 0
                dataset[i]['d1'] = 0
                dataset[i]['d2'] = 0
                dataset[i]['d3'] = 0
                dataset[i]['d4'] = 0
            elif dataset[i]['number_digits'] == 4:
                dataset[i]['d1'] = img_data['label'][0]
                dataset[i]['d2'] = img_data['label'][1]
                dataset[i]['d3'] = img_data['label'][2]
                dataset[i]['d4'] = img_data['label'][3]
            elif dataset[i]['number_digits'] == 3:
                dataset[i]['d1'] = 0
                dataset[i]['d2'] = img_data['label'][0]
                dataset[i]['d3'] = img_data['label'][1]
                dataset[i]['d4'] = img_data['label'][2]
            elif dataset[i]['number_digits'] == 2:
                dataset[i]['d1'] = 0
                dataset[i]['d2'] = 0
                dataset[i]['d3'] = img_data['label'][0]
                dataset[i]['d4'] = img_data['label'][1]
            elif dataset[i]['number_digits'] == 1:
                dataset[i]['d1'] = 0
                dataset[i]['d2'] = 0
                dataset[i]['d3'] = 0
                dataset[i]['d4'] = img_data['label'][0]
            # add negative sample
            if add_negative_samples == True:
                raw_negative_image = Image.open(os.path.join(dataset_dir, img_name)) # PIL
                # get bounding containing negative sample
                left = right + 5
                right = min([left + box_width, image_width])
                top = min([top + 5, image_height])
                bottom = top + box_height
                img = raw_negative_image.crop([left, top, right, bottom])
                img = img.resize([64, 64])
                np_img = np.array(img)
                dataset_negative_samples[number_negative_samples]['image'] = np_img
                dataset_negative_samples[number_negative_samples]['number_digits'] = 0
                dataset_negative_samples[number_negative_samples]['d1'] = 0
                dataset_negative_samples[number_negative_samples]['d2'] = 0
                dataset_negative_samples[number_negative_samples]['d3'] = 0
                dataset_negative_samples[number_negative_samples]['d4'] = 0
                number_negative_samples = number_negative_samples + 1
                #img.show()
                #cv2.imshow('image', cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
                #cv2.waitKey(0)
        dataset = dataset[(dataset['number_digits'] >= 1) & (dataset['number_digits'] <= 4)]
        if add_negative_samples == True:
            dataset_negative_samples = dataset_negative_samples[0:int(number_negative_samples/4)]
            dataset_all = np.empty(dataset.shape[0] + dataset_negative_samples.shape[0], dtype=structure)
            dataset_all[0:dataset.shape[0]] = dataset
            dataset_all[dataset.shape[0]:dataset.shape[0]+dataset_negative_samples.shape[0]] = dataset_negative_samples
            idx_rnd = np.random.permutation(dataset_all.shape[0])
            dataset_all = dataset_all[idx_rnd]
        return dataset_all

    def _stack_structured_arrays(self, arr_1, arr_2):
        image_dt = np.dtype((np.float32, (64,64,3)))
        structure = [('image', image_dt), ('number_digits', np.int32),
                     ('d1', np.int32), ('d2', np.int32),
                     ('d3', np.int32), ('d4', np.int32)]
        dataset_all = np.empty(arr_1.shape[0] + arr_2.shape[0], dtype=structure)
        dataset_all[0:arr_1.shape[0]] = arr_1
        dataset_all[arr_1.shape[0]:arr_1.shape[0]+arr_2.shape[0]] = arr_2
        return dataset_all

    def print_samples(self, data, num_samples):
        idx_rnd = np.random.randint(0, data.shape[0], num_samples)
        for i in idx_rnd:
            print('label: {} - {} - {} - {} - {}'.format(data[i]['number_digits'],
                                                         data[i]['d1'],
                                                         data[i]['d2'],
                                                         data[i]['d3'],
                                                         data[i]['d4']))
            img = Image.fromarray(data[i]['image'].astype(np.uint8))
            img.show()
            cv2.imshow('image', cv2.cvtColor(data[i]['image'].astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
