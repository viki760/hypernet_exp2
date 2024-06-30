from copy import deepcopy
import logging
from os import path
import os

from argparse import Namespace

from typing import List, Optional

import hydra
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf

import torch
import lightning as L

from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from lightning.fabric import Fabric, seed_everything

from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST
from tqdm import tqdm

from src.models import get_mnet_model, get_hnet_model
# import src.models as models

from src.datamodule import get_cifar_datamodule

logger = logging.getLogger(__name__)


torch.set_float32_matmul_precision('medium')
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torchvision
# if _TORCHVISION_AVAILABLE:
#     from torchvision import transforms

import lightning as L
import torch

from lightning.pytorch.demos import Transformer


class LightningTransformer(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)




def train(args, task_id, dataloader, task_embs, mnet, hnet):
    pass

    # param = hnet.parameters()

    # cur_embedding = xxxx

    # emb_optimizer = Adam(cur_embedding)

    # theta_optimizer = get_optimizer(param) #????

    #     # Learning rate schedulers
    # ### Prepare CL Regularizer ###

    # #     # We need to tell the main network, which batch statistics to use, in case
    # # # batchnorm is used and we checkpoint the batchnorm stats.
    # # mnet_kwargs = {}
    # # if mnet.batchnorm_layers is not None:
    # #     if config.bn_distill_stats:
    # #         raise NotImplementedError()
    # #     elif not config.bn_no_running_stats and \
    # #             not config.bn_no_stats_checkpointing:
    # #         # Specify current task as condition to select correct
    # #         # running stats.
    # #         mnet_kwargs['condition'] = task_id

    #     ######################
    # ### Start training ###
    # ######################
    prev_hnet = deepcopy(hnet)
    prev_hnet.eval()

    task_criterion = nn.CrossEntropyLoss()

    hnet_optimizer = optim.SGD(hnet.parameters())

    task_embs: List[torch.Tensor]

    for epoch in range(3):
        for images, labels in dataloader:
            
            images = images.to(args.cfg.device)
            labels = labels.to(args.cfg.device)

            print(images.shape)
            print(labels.shape)
            print(labels.unique())


            # get delta_theta (candidate update)
            current_emb = task_embs[task_id]
            with torch.no_grad():
                # trial_hnet = deepcopy(hnet)
                trial_hnet = hnet
            trial_hnet.train()
            trial_hnet_optimizer = optim.SGD(trial_hnet.parameters())

            trial_hnet_optimizer.zero_grad()

            trial_weights = trial_hnet(current_emb)
            trial_Y_hat_logits = mnet(images, weights=trial_weights) 

            trial_loss_task = task_criterion(trial_Y_hat_logits, labels)
            trial_loss_task.retain_grad()
            trial_loss_task.backward()

            trial_hnet_optimizer.step()


            # compute reg loss
            loss_reg = 0
            
            trial_hnet_optimizer.zero_grad()
            for i in range(task_id):
                weights_i_prev = prev_hnet(task_embs[i])

                weights_i = trial_hnet(task_embs[i])
                loss_reg += (weights_i - weights_i_prev).pow(2).sum()

            # update hnet
            hnet_optimizer.zero_grad()

            weights = hnet(task_embs[task_id])
            Y_hat_logits = mnet(images, weights=weights)
            loss_task = task_criterion(Y_hat_logits, labels)

            loss = loss_task + args.cfg.beta * loss_reg
            loss.retain_grad()
            loss.backward(retain_graph=True)

            for param, trial_param in zip(hnet.parameters(), trial_hnet.parameters()):
                param.grad += trial_param.grad

            hnet_optimizer.step()
            break
        break

    #     #########################
    #     # Learning rate scheduler
    #     #########################
    #     ###########################
    #     ### Tensorboard summary ###
    #     ###########################


def run(args: Namespace):
    seed_everything(args.cfg.seed)  # instead of torch.manual_seed(...)

    logger.info("test2333")

    data_module_list = get_cifar_datamodule(args)

    logger.info(f"data list len: {len(data_module_list)}") 

    mnet = get_mnet_model(args)
    hnet = get_hnet_model(args, mnet)

    task_embs = [torch.randn(args.cfg.task_emb_dim, requires_grad=True) for _ in range(args.cfg.num_tasks)]

    for idx in range(len(task_embs)):
        torch.nn.init.normal_(task_embs[idx], mean=0., std=1.)

    for i in range(args.cfg.num_tasks):
        # print(data_modules[i].prepare_data())

        dm = data_module_list[i]
        dm.prepare_data()
        dm.setup(stage="fit")

        # if hnet is not None and config.init_with_prev_emb and j > 0:
        #     last_emb = hnet.get_task_emb(j-1).detach().clone()
        #     hnet.get_task_emb(j).data = last_emb

        train(
            args=args,
            task_id=i,
            dataloader=dm.train_dataloader(),
            task_embs=task_embs,
            mnet=mnet,
            hnet=hnet
        )

# if __name__ == "__main__":
#     run()



