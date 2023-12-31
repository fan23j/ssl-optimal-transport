from torch.utils.data import Dataset
from .cifar100 import CIFAR100
from .nuswide import NUSWIDEClassification
import torch
import clip
import os
import json


class MixedDatasetNuswideCifar(Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.cifar_dataset = CIFAR100(
            cfg,
            os.path.join(root, "cifar-100-python"),
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
        self.cifar_len = len(self.cifar_dataset)
        self.total_len = self.nuswide_len + self.cifar_len
        self.epoch_counter = 0
        with open(cfg.DATASET.MIXED_LABELS, "r") as file:
            self.mixed_labels = json.load(file)

        self.mixed_indices = {value: key for key, value in self.mixed_labels.items()}

        self.multilabel_labels = {
            category: index
            for index, category in enumerate(self.nuswide_dataset.all_categories)
        }

        self.multilabel_indices = {
            index: category
            for index, category in enumerate(self.nuswide_dataset.all_categories)
        }

        self.multiclass_labels = {
            category: index for index, category in enumerate(self.cifar_dataset.classes)
        }

        self.multiclass_indices = {
            index: category for index, category in enumerate(self.cifar_dataset.classes)
        }

        multilabel_descriptions = [
            "a photo that contains a " + category
            for category in self.nuswide_dataset.all_categories
        ]

        multiclass_descriptions = [
            "a photo of a " + category for category in self.cifar_dataset.classes
        ]

        # Tokenize
        self.multilabel_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multilabel_descriptions]
        )

        self.multiclass_text_inputs = torch.cat(
            [clip.tokenize(description) for description in multiclass_descriptions]
        )

    def on_epoch_start(self):
        self.cifar_permutation = torch.randperm(self.cifar_len)
        self.nuswide_permutation = torch.randperm(self.nuswide_len)

    def __getitem__(self, index):
        # Determine dataset and retrieve item
        if index % 2 == 0:  # Choose 'nuswide'
            index = self.nuswide_permutation[index // 2 % self.nuswide_len]
            return self.nuswide_dataset[index], 0
        else:  # Choose 'cifar'
            index = self.cifar_permutation[index // 2 % self.cifar_len]
            return self.cifar_dataset[index], 1

    def __len__(self):
        return self.total_len
