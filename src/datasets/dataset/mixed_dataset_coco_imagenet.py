from torch.utils.data import Dataset
from .coco import Coco
from .cifar100 import CIFAR100
from .tiny_imagenet import TinyImageNet
import torch
import clip


class MixedDatasetCocoImageNet(Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.coco_dataset = Coco(
            cfg, os.path.join(root,"coco"), train=train, download=download, sampler=sampler
        )
        self.imagenet_dataset = TinyImageNet(
            cfg, os.path.join(root,"tiny-imagenet-200", train=train, download=download, sampler=sampler
        )
        self.train = train
        self.coco_len = len(self.coco_dataset)
        self.imagenet_len = len(self.imagenet_dataset)
        self.total_len = self.coco_len + self.imagenet_len
        self.epoch_counter = 0
        self.all_unique_categories = list(
            set(self.coco_dataset.all_categories + self.imagenet_dataset.classes)
        )

        descriptions = [
            "a photo that contains a " + category
            for category in self.all_unique_categories
        ]

        # Tokenize
        self.text_inputs = torch.cat(
            [clip.tokenize(description) for description in descriptions]
        )

    def on_epoch_start(self):
        self.coco_permutation = torch.randperm(self.coco_len)
        self.imagenet_permutation = torch.randperm(self.imagenet_len)

    def __getitem__(self, index):
        # Determine dataset and retrieve item
        if index % 2 == 0:  # Choose 'coco'
            index = self.coco_permutation[index // 2 % self.coco_len]
            return self.coco_dataset[index], 0
        else:  # Choose 'imagenet'
            index = self.imagenet_permutation[index // 2 % self.imagenet_len]
            return self.imagenet_dataset[index], 1

    def __len__(self):
        return self.total_len
