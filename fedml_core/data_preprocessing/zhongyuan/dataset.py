import torch
from torch.utils.data import Dataset


class zhongyuan_dataset(Dataset):
    def __init__(self, data):
        super(zhongyuan_dataset, self).__init__()
        self.Xa_data = data[0]
        self.Xb_data = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (self.Xa_data[index], self.Xb_data[index])
        Y = self.labels[index]
        return X, Y

