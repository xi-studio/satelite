import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
import time
from skimage.transform import rescale, resize

class Sate(Dataset):
    def __init__(self, filenames, flag=True):
        super(Sate, self).__init__()

        self.list = filenames
        self.flag = flag

    def preprocess(self, x):
        x[0] = (x[0] - 220.0) / (315.0 - 220.0)
        x[1] = (x[1]/100.0 - 950.0) / (1050.0 - 950.0)
        x[2] = (x[2] - (-30.0)) / (30.0 - (-30.0))
        x[3] = (x[3] - (-30.0)) / (30.0 - (-30.0))

        return x


    def __getitem__(self, index):

        if self.flag:
            x = np.ones((10, 256, 256), dtype=np.float32)
            y = np.ones((256, 256), dtype=np.float32)
        else:
            sate = np.load(self.list[index][0][1:]).astype(np.float32)
            pred = np.load(self.list[index][1][1:]).astype(np.float32)
            obs  = np.load(self.list[index][2][1:]).astype(np.float32)

            pred = pred[(3, 0, 1, 2), :, :]

            sate = np.nan_to_num(sate, nan=255)
            sate = (sate - 180.0) / (375.0 - 180.0)

            pred = self.preprocess(pred)
            obs  = self.preprocess(obs)

            sate = resize(sate, (10, 256, 256))
            pred = resize(pred, (4, 256, 256))
            obs  = resize(obs, (4, 256, 256))

            pred_input  = np.concatenate((sate, pred), axis=0)

            x = pred_input[:13]
            y = obs[0]

        return x, y 

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    filenames = np.load('data/meta/train_pangu_24.npy')
    dataset = Sate(filenames, flag=False)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
    for x in train_loader:
        a,b = x
        print(a.shape, b.shape)
        print(len(x))
        break
