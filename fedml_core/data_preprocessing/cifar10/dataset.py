import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

class IndexedCIFAR10(CIFAR10):
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


class split_dataset(Dataset):
    def __init__(self, data):
        super(split_dataset, self).__init__()
        self.Xa_data = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.Xa_data[index]
        Y = self.labels[index]
        return X, Y



class cluster_dataset(CIFAR10):
    def __init__(self, *args, new_targets=None, **kwargs):
        super().__init__(*args, **kwargs)

        # 如果提供了新的标签，那么就替换原始标签
        if new_targets is not None:
            assert len(new_targets) == len(self.targets)
            self.targets = new_targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
