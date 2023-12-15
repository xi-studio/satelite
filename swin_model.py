import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from einops import rearrange

import lightning.pytorch as pl
from models.swin_transformer import SwinTransformer3D


class SwModel(pl.LightningModule):
    def __init__(self, config):
        super(SwModel, self).__init__()
        self.swin = SwinTransformer3D(in_chans=5,
                                      embed_dim=96 * 2
                                     ) 

        self.conv = nn.Conv2d(96 * 2 * 8, 1024, kernel_size=3, padding=1)

    def forward(self, x):
        x = rearrange(x, 'n (c d) h w -> n c d h w', c=5)

        x = self.swin(x)

        x = rearrange(x, 'n c d h w -> n (c d) h w')

        x = self.conv(x)

        x = rearrange(x, 'n (c c1) h w -> n (c h) (c1 w)', c1=32)
        
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        yp = self.forward(x)
        loss = F.l1_loss(yp, y) 
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        yp = self.forward(x)
        loss = F.l1_loss(yp, y)
        self.log('val_loss', loss)
        #if batch_nb == 0:
        #    name = 'data/predict_sample/tp_uv_sample/img_%05d.png' % self.current_epoch
        #    save_image(y_hat.cpu(), name)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

