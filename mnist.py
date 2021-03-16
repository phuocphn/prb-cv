import argparse
import os
import importlib
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class LiMMNIST(pl.LightningModule):
    
    def __init__(self, data_dir='~/data',
                    arch='arch1', 
                    learning_rate=1.0, 
                    gamma=0.7, 
                    num_workers=2, batch_size=64):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = os.path.expanduser(data_dir)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        module = importlib.import_module("models.mnist")
        self.model = getattr(module, arch)()
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.logger.experiment.add_scalar('lr', self.trainer.optimizers[0].param_groups[0]['lr'], self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y) 

        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y) 
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate, )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]


    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)


    def setup(self, stage=None):
        self.trainset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.valset = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

