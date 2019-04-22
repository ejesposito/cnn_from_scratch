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
                                                           self.PREPROCESSED_DIR, file_name='train.npy',
                                                           add_negative_samples = True)
        else:
            print('Loading train.npy')
            self.train = np.load(self.TRAIN_NPY)
        # load or create test dataset
        test_file = Path(self.TEST_NPY)
        if not test_file.is_file():
            print('Creating test.npy')
            self.test = self._create_preprocessed_dataset(self.TEST_DATASET_DIR,
                                                          self.PREPROCESSED_DIR, file_name='test.npy',
                                                          add_negative_samples = False)
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

    def _create_preprocessed_dataset(self, dataset_dir, preprocessed_dir, file_name, add_negative_samples = False):
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
            #v2.waitKey(0)
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
            # ADD MY IMAGES
            img1 = Image.open(os.path.join(dataset_dir, '1035_1.jpg')) # PIL
            img2 = Image.open(os.path.join(dataset_dir, '1035_2.jpg')) # PIL
            img3 = Image.open(os.path.join(dataset_dir, '1035_3.jpg')) # PIL
            img4 = Image.open(os.path.join(dataset_dir, '1035_4.jpg')) # PIL
            img5 = Image.open(os.path.join(dataset_dir, '1035_5.jpg')) # PIL
            img6 = Image.open(os.path.join(dataset_dir, '331_1.jpg')) # PIL
            img7 = Image.open(os.path.join(dataset_dir, '331_2.jpg')) # PIL
            img8 = Image.open(os.path.join(dataset_dir, '261_1.jpg')) # PIL
            img9 = Image.open(os.path.join(dataset_dir, '259_1.jpg')) # PIL
            img10 = Image.open(os.path.join(dataset_dir, '1036_1.jpg')) # PIL
            img11 = Image.open(os.path.join(dataset_dir, '1036_2.jpg')) # PIL
            img12 = Image.open(os.path.join(dataset_dir, '1036_3.jpg')) # PIL
            img13 = Image.open(os.path.join(dataset_dir, '1036_4.jpg')) # PIL
            img14 = Image.open(os.path.join(dataset_dir, '1036_5.jpg')) # PIL
            img15 = Image.open(os.path.join(dataset_dir, '1036_6.jpg')) # PIL
            img16 = Image.open(os.path.join(dataset_dir, '1036_7.jpg')) # PIL
            img17 = Image.open(os.path.join(dataset_dir, '1036_8.jpg')) # PIL
            img18 = Image.open(os.path.join(dataset_dir, '1036_9.jpg')) # PIL
            img19 = Image.open(os.path.join(dataset_dir, '1036_10.jpg')) # PIL
            img20 = Image.open(os.path.join(dataset_dir, '1036_11.jpg')) # PIL
            img1 = img1.resize([64, 64])
            img2 = img2.resize([64, 64])
            img3 = img3.resize([64, 64])
            img4 = img4.resize([64, 64])
            img5 = img5.resize([64, 64])
            img6 = img6.resize([64, 64])
            img7 = img7.resize([64, 64])
            img8 = img8.resize([64, 64])
            img9 = img9.resize([64, 64])
            img10 = img10.resize([64, 64])
            img11 = img11.resize([64, 64])
            img12 = img12.resize([64, 64])
            img13 = img13.resize([64, 64])
            img14 = img14.resize([64, 64])
            img15 = img15.resize([64, 64])
            img16 = img16.resize([64, 64])
            img17 = img17.resize([64, 64])
            img18 = img18.resize([64, 64])
            img19 = img19.resize([64, 64])
            img20 = img20.resize([64, 64])
            extra_dataset = np.empty(20, dtype=structure)
            extra_dataset[0]['image'] = img1 # 1035
            extra_dataset[0]['number_digits'] = 4
            extra_dataset[0]['d1'] = 1
            extra_dataset[0]['d2'] = 10
            extra_dataset[0]['d3'] = 3
            extra_dataset[0]['d4'] = 5
            extra_dataset[1]['image'] = img2 # 1035
            extra_dataset[1]['number_digits'] = 4
            extra_dataset[1]['d1'] = 1
            extra_dataset[1]['d2'] = 10
            extra_dataset[1]['d3'] = 3
            extra_dataset[1]['d4'] = 5
            extra_dataset[2]['image'] = img3 # 1035
            extra_dataset[2]['number_digits'] = 4
            extra_dataset[2]['d1'] = 1
            extra_dataset[2]['d2'] = 10
            extra_dataset[2]['d3'] = 3
            extra_dataset[2]['d4'] = 5
            extra_dataset[3]['image'] = img4 # 1035
            extra_dataset[3]['number_digits'] = 4
            extra_dataset[3]['d1'] = 1
            extra_dataset[3]['d2'] = 10
            extra_dataset[3]['d3'] = 3
            extra_dataset[3]['d4'] = 5
            extra_dataset[4]['image'] = img5 # 1035
            extra_dataset[4]['number_digits'] = 4
            extra_dataset[4]['d1'] = 1
            extra_dataset[4]['d2'] = 10
            extra_dataset[4]['d3'] = 3
            extra_dataset[4]['d4'] = 5
            extra_dataset[5]['image'] = img6 # 331
            extra_dataset[5]['number_digits'] = 3
            extra_dataset[5]['d1'] = 0
            extra_dataset[5]['d2'] = 3
            extra_dataset[5]['d3'] = 3
            extra_dataset[5]['d4'] = 1
            extra_dataset[6]['image'] = img7 # 331
            extra_dataset[6]['number_digits'] = 3
            extra_dataset[6]['d1'] = 0
            extra_dataset[6]['d2'] = 3
            extra_dataset[6]['d3'] = 3
            extra_dataset[6]['d4'] = 1
            extra_dataset[7]['image'] = img8 # 261
            extra_dataset[7]['number_digits'] = 3
            extra_dataset[7]['d1'] = 0
            extra_dataset[7]['d2'] = 2
            extra_dataset[7]['d3'] = 6
            extra_dataset[7]['d4'] = 1
            extra_dataset[8]['image'] = img9 # 259
            extra_dataset[8]['number_digits'] = 3
            extra_dataset[8]['d1'] = 0
            extra_dataset[8]['d2'] = 2
            extra_dataset[8]['d3'] = 5
            extra_dataset[8]['d4'] = 9
            extra_dataset[9]['image'] = img10 # 1036
            extra_dataset[9]['number_digits'] = 4
            extra_dataset[9]['d1'] = 1
            extra_dataset[9]['d2'] = 10
            extra_dataset[9]['d3'] = 3
            extra_dataset[9]['d4'] = 6
            extra_dataset[10]['image'] = img11
            extra_dataset[10]['number_digits'] = 4
            extra_dataset[10]['d1'] = 1
            extra_dataset[10]['d2'] = 10
            extra_dataset[10]['d3'] = 3
            extra_dataset[10]['d4'] = 6
            extra_dataset[11]['image'] = img12
            extra_dataset[11]['number_digits'] = 4
            extra_dataset[11]['d1'] = 1
            extra_dataset[11]['d2'] = 10
            extra_dataset[11]['d3'] = 3
            extra_dataset[11]['d4'] = 6
            extra_dataset[12]['image'] = img13
            extra_dataset[12]['number_digits'] = 4
            extra_dataset[12]['d1'] = 1
            extra_dataset[12]['d2'] = 0
            extra_dataset[12]['d3'] = 3
            extra_dataset[12]['d4'] = 6
            extra_dataset[13]['image'] = img14
            extra_dataset[13]['number_digits'] = 4
            extra_dataset[13]['d1'] = 1
            extra_dataset[13]['d2'] = 10
            extra_dataset[13]['d3'] = 3
            extra_dataset[13]['d4'] = 6
            extra_dataset[14]['image'] = img15
            extra_dataset[14]['number_digits'] = 4
            extra_dataset[14]['d1'] = 1
            extra_dataset[14]['d2'] = 10
            extra_dataset[14]['d3'] = 3
            extra_dataset[14]['d4'] = 6
            extra_dataset[15]['image'] = img16
            extra_dataset[15]['number_digits'] = 4
            extra_dataset[15]['d1'] = 1
            extra_dataset[15]['d2'] = 10
            extra_dataset[15]['d3'] = 3
            extra_dataset[15]['d4'] = 6
            extra_dataset[16]['image'] = img17
            extra_dataset[16]['number_digits'] = 4
            extra_dataset[16]['d1'] = 1
            extra_dataset[16]['d2'] = 10
            extra_dataset[16]['d3'] = 3
            extra_dataset[16]['d4'] = 6
            extra_dataset[17]['image'] = img18
            extra_dataset[17]['number_digits'] = 4
            extra_dataset[17]['d1'] = 1
            extra_dataset[17]['d2'] = 10
            extra_dataset[17]['d3'] = 3
            extra_dataset[17]['d4'] = 6
            extra_dataset[18]['image'] = img19
            extra_dataset[18]['number_digits'] = 4
            extra_dataset[18]['d1'] = 1
            extra_dataset[18]['d2'] = 10
            extra_dataset[18]['d3'] = 3
            extra_dataset[18]['d4'] = 6
            extra_dataset[19]['image'] = img20
            extra_dataset[19]['number_digits'] = 4
            extra_dataset[19]['d1'] = 1
            extra_dataset[19]['d2'] = 10
            extra_dataset[19]['d3'] = 3
            extra_dataset[19]['d4'] = 6
            # BUILD FINAL DATASET
            dataset_final = np.empty(dataset_all.shape[0] + extra_dataset.shape[0], dtype=structure)
            dataset_final[0:dataset_all.shape[0]] = dataset_all
            dataset_final[dataset_all.shape[0]:dataset_all.shape[0]+extra_dataset.shape[0]] = extra_dataset
            dataset = dataset_final
            idx_rnd = np.random.permutation(dataset.shape[0])
            dataset = dataset[idx_rnd]
        np.save(os.path.join(preprocessed_dir, file_name), dataset)
        return dataset

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
