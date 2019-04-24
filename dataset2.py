import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TorchDataSet(Dataset):

    def __init__(self, data, target, transform):
        # create the tensors
        number_digits = target['number_digits']
        d1 = target['d1']
        d2 = target['d2']
        d3 = target['d3']
        d4 = target['d4']
        self.data = data #torch.from_numpy(data).float()
        self.number_digits = torch.from_numpy(number_digits).long()
        self.d1 = torch.from_numpy(d1).long()
        self.d2 = torch.from_numpy(d2).long()
        self.d3 = torch.from_numpy(d3).long()
        self.d4 = torch.from_numpy(d4).long()
        self._transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        img = Image.fromarray(data.astype(np.uint8))
        x = self._transform(img)
        nd = self.number_digits[index]
        d1 = self.d1[index]
        d2 = self.d2[index]
        d3 = self.d3[index]
        d4 = self.d4[index]
        return x, nd, [d1, d2, d3, d4]

    def __len__(self):
        return len(self.data)
