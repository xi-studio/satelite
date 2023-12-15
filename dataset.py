import torch
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, utils
import glob
import csv
import time
import io

class Sate(Dataset):
    def __init__(self, filenames):
        super(Sate, self).__init__()

        self.filenames = filenames


    def __getitem__(self, index):
        x = np.ones((10, 256, 256), dtype=np.float32)
        y = np.ones((256, 256), dtype=np.float32)


        return x, y 

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    dataset = Sate(np.arange(100))
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
    for x in train_loader:
        a,b = x
        print(a.shape)
        print(len(x))
        break
