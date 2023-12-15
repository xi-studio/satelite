import os
import argparse
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, utils

from swin_model import SwModel
from dataset import Sate

def main():
    filenames = np.arange(100)

    dataset = Sate(filenames)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    config = {'lr': 0.05}
    model = SwModel(config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=2, enable_checkpointing=True)
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
