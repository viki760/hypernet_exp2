import lightning as L
import torch

from .data.cifar10 import CIFAR10DataModule
from .data.cifar100 import SplitCIFAR100DataModule

def get_cifar_datamodule(args):
    batch_size = args.cfg.batch_size

    assert(1 <= args.cfg.num_tasks and args.cfg.num_tasks <= 11)

    data_list = []
    data_list.append(
        CIFAR10DataModule(
            data_dir=args.cfg.data_root_dir,
            batch_size=batch_size, 
        )
    )
    for i in range(1, args.cfg.num_tasks):
        task_index = i - 1
        print(list(range(task_index*10, (task_index+1)*10 - 3)))
        data_list.append(
            SplitCIFAR100DataModule(
                lables=list(range(task_index*10, (task_index+1)*10)), 
                data_dir=args.cfg.data_root_dir,
                batch_size=batch_size, 
            )
        )
    return data_list