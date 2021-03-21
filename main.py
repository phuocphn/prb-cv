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

from mnist import LiMMNIST
from cifar10 import LiMCIFAR10
from cifar100 import LiMCIFAR100
from tinyimagenet import LiMTinyImageNet
from tinyimagenet224 import LiMTinyImageNet224
from imagenette import LiMImagenette
from imagenet import LiMImagenet

from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.core.lightning import LightningModule

class LSQCheckpointConnector(CheckpointConnector):
    def __init__(self, *args, **kwargs):
        super(LSQCheckpointConnector, self).__init__(*args, **kwargs)
    def restore_model_state(self, model: LightningModule, checkpoint) -> None:
        """
        Restore model states from a 'PyTorch-Lightning checkpoint' dictionary object
        """

        # restore datamodule states
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.on_load_checkpoint(checkpoint)

        # hook: give user access to checkpoint if needed.
        model.on_load_checkpoint(checkpoint)

        # restore model state_dict
        model.load_state_dict(checkpoint['state_dict'])



def main(hparams):
    if hparams.dataset == 'mnist':
        model = LiMMNIST(arch=hparams.arch,
            gamma=hparams.gamma,
            learning_rate=hparams.lr, 
            batch_size=hparams.batch_size, )

    elif hparams.dataset == 'cifar10':
        model = LiMCIFAR10(arch=hparams.arch,
            learning_rate=hparams.lr, 
            weight_decay=hparams.weight_decay, 
            momentum=hparams.momentum, 
            batch_size=hparams.batch_size, )

    elif hparams.dataset == 'cifar100':
        model = LiMCIFAR100(hparams=vars(hparams))

        # model = LiMCIFAR100(arch=hparams.arch,
        #     learning_rate=hparams.lr, 
        #     weight_decay=hparams.weight_decay, 
        #     momentum=hparams.momentum, 
        #     batch_size=hparams.batch_size, )
    elif hparams.dataset == 'tinyimagenet':
        model = LiMTinyImageNet(arch=hparams.arch,
            learning_rate=hparams.lr, 
            weight_decay=hparams.weight_decay, 
            momentum=hparams.momentum, 
            batch_size=hparams.batch_size, ) 

    elif hparams.dataset == 'tinyimagenet224':
        model = LiMTinyImageNet224(arch=hparams.arch,
            learning_rate=hparams.lr, 
            weight_decay=hparams.weight_decay, 
            momentum=hparams.momentum, 
            batch_size=hparams.batch_size, ) 
    elif hparams.dataset == 'imagenette':
        model = LiMImagenette(arch=hparams.arch,
            learning_rate=hparams.lr, 
            weight_decay=hparams.weight_decay, 
            momentum=hparams.momentum, 
            batch_size=hparams.batch_size, ) 
    elif hparams.dataset == 'imagenet':
        model = LiMImagenet(arch=hparams.arch,
            learning_rate=hparams.lr, 
            weight_decay=hparams.weight_decay, 
            momentum=hparams.momentum, 
            batch_size=hparams.batch_size, ) 


    logger = TestTubeLogger("tb_logs", 
        name=hparams.expr_name, 
        description=hparams.expr_desc,
        create_git_tag=True)

    logger.experiment.tag(vars(hparams)) 
    logger.experiment.tag({'_model': str(model)})
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(gpus=hparams.gpus, 
            logger=logger,
            # callbacks=[checkpoint_callback],
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=hparams.resume,
            max_epochs=hparams.epochs, 
            deterministic=hparams.deterministic, 
            progress_bar_refresh_rate=20, 
            distributed_backend=hparams.distributed_backend, 
            weights_summary='full')
    
    trainer.checkpoint_connector = LSQCheckpointConnector(trainer)

    if hparams.evaluate:
        trainer.test()
        return 

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Hyperparams')
    parser.add_argument('--expr_name', default='cifar_10', type=str, help='the name of experiment' )
    parser.add_argument('--expr_desc', default='The detail description of the experiment', 
                        type=str, help='The detail description of the experiment' )
    parser.add_argument('--dataset', default='cifar10', 
                        choices=['cifar10', 'cifar100', 'mnist', 'tinyimagenet', 'tinyimagenet224', 'imagenette', 'imagenet'],
                        type=str, help='dataset name' )
    parser.add_argument('--train_scheme', default='fp32', 
                        choices=['fp32', 'lsq', 'uniq', 'x', 'y', 'z'],
                        type=str, help='training scheme' )

    parser.add_argument('--arch', default='ResNet18', type=str, help='network architecture.' )
    parser.add_argument('--distributed_backend', default='dp', type=str, help='distributed backend.' )

    parser.add_argument('--gpus', default=4, type=int, help='number of GPUs to train on' )
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                    help='enable deterministic training')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--bit', default=4, type=int,
                        help='quantization bit-width', dest='bit')

    args = parser.parse_args()

    main(args)

