import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, Subset

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR100
from torchvision import transforms


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=True, download=True)
        # CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar100 = CIFAR100(self.data_dir, train=True, transform=self.transform)
            self.cifar100_train, self.cifar100_val = random_split(
                cifar100, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size)


class SplitCIFAR100DataModule(CIFAR100DataModule):
    def __init__(self, lables, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = lables

    def setup(self, stage: str):
        if stage == "fit":
            cifar100 = CIFAR100(self.data_dir, train=True, transform=self.transform)

            indexes = [i for i, (label) in enumerate(cifar100.targets) if label in self.labels]

            # filtered_cifar100 = Subset(cifar100, indexes)
            filtered_cifar100 = cifar100
            filtered_cifar100.data = [filtered_cifar100.data[i] for i in indexes]
            filtered_cifar100.targets = [filtered_cifar100.targets[i] for i in indexes]

            self.cifar100_train, self.cifar100_val = random_split(
                filtered_cifar100, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )