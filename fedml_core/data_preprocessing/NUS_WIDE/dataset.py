import torch
from torch.utils.data import Dataset


class nuswide_dataset(Dataset):
    def __init__(self, data):
        super(nuswide_dataset, self).__init__()
        self.Xa_data = data[0]
        self.Xb_data = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (self.Xa_data[index], self.Xb_data[index])

        Y = self.labels[index]

        return X, Y, index

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

class cluster_dataset(Dataset):
    def __init__(self, features, labels):
        super(cluster_dataset, self).__init__()
        self.Xb_data = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #X = (self.Xa_data[index], self.Xb_data[index])
        X = self.Xb_data[index]
        Y = self.labels[index]
        return X, Y