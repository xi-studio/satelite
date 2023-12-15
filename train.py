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
    filenames = np.load('data/meta/train_pangu_24.npy')

    dataset = Sate(filenames, flag=False)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    config = {'lr': 0.05}
    model = SwModel(config)

    logger = pl.loggers.TensorBoardLogger(
                save_dir='.',
                version='sat_1215',
                name='lightning_logs'
            )
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=1, 
                         max_epochs=2, 
                         enable_checkpointing=True,
                         logger=logger
                         )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
