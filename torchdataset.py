import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TorchDataSet(Dataset):

    def __init__(self, data, target, transform=None):
        # copy arrays to avoid strides errors
        data = np.copy(data)
        number_digits = np.copy(target['number_digits'])
        d1 = np.copy(target['d1'])
        d2 = np.copy(target['d2'])
        d3 = np.copy(target['d3'])
        d4 = np.copy(target['d4'])
        # create the tensors
        self.data = data #torch.from_numpy(data).float()
        self.number_digits = torch.from_numpy(number_digits).long()
        self.d1 = torch.from_numpy(d1).long()
        self.d2 = torch.from_numpy(d2).long()
        self.d3 = torch.from_numpy(d3).long()
        self.d4 = torch.from_numpy(d4).long()
        self._transform = transform
        # move model to GPU if CUDA is available (all the dataset)
        #print('antess')
        #if torch.cuda.is_available():
        #    self.data = self.data.cuda(1)
        #    self.number_digits = self.number_digits.cuda(1)
        #    print('despues de la data')
        #    self.d1 = self.d1.cuda(1)
        #    self.d2 = self.d2.cuda(1)
        #    self.d3 = self.d3.cuda(1)
        #    self.d4 = self.d4.cuda(1)
        #    print('despues')

    def __getitem__(self, index):
        data = self.data[index]
        img = Image.fromarray(data.astype(np.uint8))
        x = self._transform(img)
        nd = self.number_digits[index]
        d1 = self.d1[index]
        d2 = self.d2[index]
        d3 = self.d3[index]
        d4 = self.d4[index]
        return x, (nd, d1, d2, d3, d4)

    def __len__(self):
        return len(self.data)
