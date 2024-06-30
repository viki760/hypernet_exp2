import logging
from os import path
import os

from argparse import Namespace

from typing import Optional

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




def train(task_id, dataloader, task_embs, mnet, hnet):
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

    for epoch in range(3):
        for images, labels in dataloader:

            print(images.shape)
            print(labels.shape)
            print(labels.unique())

            # break

            weights = hnet(task_embs[task_id])
            Y_hat_logits = mnet(images, weights=weights)
            print(Y_hat_logits.shape)
            break

        # # if config.soft_targets:

        #     task_criterion = nn.CrossEntropyLoss()

        #     loss_task = task_criterion(Y_hat_logits, labels)

            loss_task.backward(retain_graph=calc_reg, create_graph=calc_reg and \
                           config.backprop_dt)
                
            # The current task embedding only depends in the task loss, so we can
            # update it already.
            if emb_optimizer is not None:
                emb_optimizer.step()

        #     dTheta = opstep.calc_delta_theta(theta_optimizer, False,
        #         lr=config.lr, detach_dt=not config.backprop_dt)

        #     if config.continue_emb_training:
        #         dTembs = dTheta[-task_id:]
        #         dTheta = dTheta[:-task_id]
        #     else:
        #         dTembs = None

        #     loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
        #         targets=targets_hypernet, dTheta=dTheta, dTembs=dTembs,
        #         mnet=mnet, inds_of_out_heads=regged_outputs,
        #         prev_theta=prev_theta, prev_task_embs=prev_task_embs,
        #         batch_size=config.cl_reg_batch_size)

            loss_reg *= config.beta

            loss_reg.backward()

        #     # Now that we computed the regularizer, we can use the accumulated
        #     # gradients and update the hnet (or mnet) parameters.
        #     theta_optimizer.step()

        #     Y_hat = F.softmax(Y_hat_logits, dim=1)
        #     classifier_accuracy = Classifier.accuracy(Y_hat, T) * 100.0


    #     #########################
    #     # Learning rate scheduler
    #     #########################
    #     ###########################
    #     ### Tensorboard summary ###
    #     ###########################
    #         break
    #     break


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
            task_id=i,
            dataloader=dm.train_dataloader(),
            task_embs=task_embs,
            mnet=mnet,
            hnet=hnet
        )

# if __name__ == "__main__":
#     run()



