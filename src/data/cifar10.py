import lightning as L
import torch
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        # CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar10 = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(
                cifar10, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
            )

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        # if stage == "predict":
        #     self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.cifar10_test, batch_size=32)

    # def predict_dataloader(self):
    #     return DataLoader(self.cifar10_predict, batch_size=32)