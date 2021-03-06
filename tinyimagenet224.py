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


class LiMTinyImageNet224(pl.LightningModule):
    #https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_224.ipynb
    def __init__(self, data_dir='~/data',
                    arch='resnet18', 
                    learning_rate=0.001, 
                    momentum=0.9,
                    weight_decay=5e-4, 
                    num_workers=2, batch_size=128):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = os.path.expanduser(data_dir)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Hardcode some dataset specific attributes
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        module = importlib.import_module("models.tinyimagenet224")
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
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]


    def prepare_data(self):
        pass


    def setup(self, stage=None):
        self.trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'tiny-imagenet-200-224', 'train'), 
            transform=self.data_transforms['train'])

        self.valset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'tiny-imagenet-200-224', 'val'), 
            transform=self.data_transforms['val'])

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)