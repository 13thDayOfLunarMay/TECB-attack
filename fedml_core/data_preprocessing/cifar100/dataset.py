import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np

class IndexedCIFAR100(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)



        return img, target, index
