"""Define dataset class."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#####from model1.model1.m1_hps.load_data_pytorch import load_data
from load_data_pytorch import load_data

class dataset(Dataset):
    """Class for dataset."""

    def __init__(self, x, y):
        """Initialize dataset class."""
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        """Get item from dataset."""
        return self.x[idx], self.y[idx]

    def __len__(self):
        """Get length of dataset."""
        return self.length


if __name__ == '__main__':
    config = {
        'proportion': 0.90,         # A value between [0., 1.] indicating how to split data between
                                    # training set and validation set. `prop` corresponds to the
                                    # ratio of data in training set. `1.-prop` corresponds to the
                                    # amount of data in validation set.
        'print shape': 0            # Print the data shape.
    }

    (train_X, train_y), (valid_X, valid_y) = load_data(config)

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')

    trainset = dataset(train_X, train_y)

    #DataLoader
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)