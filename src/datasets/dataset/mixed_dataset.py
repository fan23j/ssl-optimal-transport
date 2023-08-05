from torch.utils.data import Dataset
from .coco import Coco
from .cifar100 import CIFAR100
import torch


class MixedDataset(Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.coco_dataset = Coco(
            cfg, root, train=train, download=download, sampler=sampler
        )
        self.cifar_dataset = CIFAR100(
            cfg, root, train=train, download=download, sampler=sampler
        )
        self.train = train
        self.coco_len = len(self.coco_dataset)
        self.cifar_len = len(self.cifar_dataset)
        self.total_len = self.coco_len + self.cifar_len
        self.epoch_counter = 0

    def on_epoch_start(self):
        self.coco_permutation = torch.randperm(self.coco_len)
        self.cifar_permutation = torch.randperm(self.cifar_len)

    def __getitem__(self, index):
        # Determine dataset and retrieve item
        if index % 2 == 0:  # Choose 'coco'
            index = self.coco_permutation[index // 2 % self.coco_len]
            return self.coco_dataset[index], 0
        else:  # Choose 'cifar'
            index = self.cifar_permutation[index // 2 % self.cifar_len]
            return self.cifar_dataset[index], 1

    def __len__(self):
        return self.total_len
