from torch.utils.data import Dataset
from .tiny_imagenet import TinyImageNet
from .nuswide import NUSWIDEClassification
import torch
import clip
import os


class MixedDatasetNuswideImageNet(Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.imagenet_dataset = TinyImageNet(
            cfg,
            os.path.join(root, "tiny-imagenet-200"),
            train=train,
            download=download,
            sampler=sampler,
        )
        self.nuswide_dataset = NUSWIDEClassification(
            cfg,
            os.path.join(root, "nuswide"),
            train=train,
            download=download,
            sampler=sampler,
        )
        self.train = train
        self.nuswide_len = len(self.nuswide_dataset)
        self.imagenet_len = len(self.imagenet_dataset)
        self.total_len = self.nuswide_len + self.imagenet_len
        self.epoch_counter = 0
        self.all_unique_categories = list(
            set(
                self.nuswide_dataset.all_categories + self.imagenet_dataset.class_labels
            )
        )
        self.mixed_labels = {
            category: index for index, category in enumerate(self.all_unique_categories)
        }

        descriptions = [
            "a photo that contains a " + category
            for category in self.all_unique_categories
        ]

        # Tokenize
        self.text_inputs = torch.cat(
            [clip.tokenize(description) for description in descriptions]
        )

    def on_epoch_start(self):
        self.imagenet_permutation = torch.randperm(self.imagenet_len)
        self.nuswide_permutation = torch.randperm(self.nuswide_len)

    def __getitem__(self, index):
        # Determine dataset and retrieve item
        if index % 2 == 0:  # Choose 'nuswide'
            index = self.nuswide_permutation[index // 2 % self.nuswide_len]
            return self.nuswide_dataset[index], 0
        else:  # Choose 'imagenet'
            index = self.imagenet_permutation[index // 2 % self.imagenet_len]
            return self.imagenet_dataset[index], 1

    def __len__(self):
        return self.total_len
