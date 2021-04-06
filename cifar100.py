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


class LiMCIFAR100(pl.LightningModule):
    
    def __init__(self, hparams):
                    #data_dir='~/data',
                    #arch='ResNet18', 
                    #learning_rate=0.1, 
                    #momentum=0.9,
                    #weight_decay=5e-4, 
                    #num_workers=2, batch_size=128, *args, **kwargs):
        super().__init__()

        # Set our init args as class attributes
        arch = hparams.get("arch")
        train_scheme = hparams.get("train_scheme",)

        self.train_scheme = train_scheme
        self.data_dir = os.path.expanduser(hparams.get("data_dir", "~/data"))
        self.learning_rate = hparams.get("lr")
        self.weight_decay = hparams.get("weight_decay")
        self.momentum = hparams.get("momentum")
        self.batch_size = hparams.get("batch_size")
        self.num_workers = hparams.get("num_workers", 4)

        # Hardcode some dataset specific attributes
        self.num_classes = 10

        self.transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])

        if train_scheme == "lsq":
            module = importlib.import_module("models.cifar100.lsq_quan_models")
            bit = hparams.get("bit")
            self.model = getattr(module, arch)(bit=bit)
        elif train_scheme == "sw_precision":
            module = importlib.import_module("models.cifar100.sw_precision_models")
            self.model = getattr(module, arch)()
            self.model.current_bit = 8
        elif train_scheme == "fp32":
            module = importlib.import_module("models.cifar100")
            self.model = getattr(module, arch)()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.logger.experiment.add_scalar('lr', self.trainer.optimizers[0].param_groups[0]['lr'], self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y) 

        # Tensorboard logs 
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y) 
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        if self.train_scheme == "sw_precision":
            self.log('acc_' + str(self.model.current_bit), acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]


    def prepare_data(self):
        torchvision.datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=self.transform_train)
        torchvision.datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=self.transform_test)


    def setup(self, stage=None):
        self.trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform_train)

        self.valset = torchvision.datasets.CIFAR100(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform_test)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)