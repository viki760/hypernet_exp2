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

logger = logging.getLogger(__name__)


torch.set_float32_matmul_precision('medium')
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torchvision
# if _TORCHVISION_AVAILABLE:
#     from torchvision import transforms


from main import run

# https://stackoverflow.com/questions/35155382/copying-specific-files-to-a-new-folder-while-maintaining-the-original-subdirect
def include_patterns(*patterns):
    import fnmatch
    from os.path import isdir, join
    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                        for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names.
        # dir_names = (name for name in all_names if isdir(join(path, name)))
        dir_names = []
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns


def backup_source_code():
    import shutil
    from hydra.utils import get_original_cwd

    shutil.copytree(get_original_cwd(), "./_code_backup", 
                    ignore=include_patterns("src", "conf", "*.py", "*.yaml"), copy_function=shutil.copy)
    # exit(0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    from glob import has_magic
    from hydra.core.hydra_config import HydraConfig
    if has_magic(HydraConfig.get().runtime.output_dir):
        raise ValueError("output_dir should not contain any magic characters")

    backup_source_code()

    args = Namespace()
    args.cfg = cfg

    # with logging_redirect_tqdm(loggers=[logger]):
    with logging_redirect_tqdm(): # It's the root logger that Hydra is redirecting
        run(args)

if __name__ == "__main__":

    main()