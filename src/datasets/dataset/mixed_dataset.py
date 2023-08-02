from torch.utils.data import Dataset
from .coco import Coco
from .cifar100 import CIFAR100


class MixedDataset(Dataset):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        eval_dataset = "mixed"
        train_ratio = 2.36
        val_ratio = 0.5
        self.ratio = train_ratio if train else val_ratio
        self.coco_dataset = Coco(
            cfg, root, train=train, download=download, sampler=sampler
        )
        self.cifar_dataset = CIFAR100(
            cfg, root, train=train, download=download, sampler=sampler
        )
        self.eval_dataset = eval_dataset
        self.train = train

    def __getitem__(self, index):
        if self.train:
            if (
                index % (self.ratio + 1) < 1
            ):  # If this index falls within the CIFAR "slice"
                return self.cifar_dataset[
                    int(index / (self.ratio + 1)) % len(self.cifar_dataset)
                ]
            else:  # If this index falls within the COCO "slice"
                return self.coco_dataset[
                    int(index / (self.ratio + 1)) % len(self.coco_dataset)
                ]
        else:
            if self.eval_dataset == "coco":
                return self.coco_dataset[index % len(self.coco_dataset)]
            elif self.eval_dataset == "cifar":
                return self.cifar_dataset[index % len(self.cifar_dataset)]
            else:  # Mixed sampling during evaluation
                if (
                    index % (self.ratio + 1) < 1
                ):  # If this index falls within the CIFAR "slice"
                    return self.cifar_dataset[
                        int(index / (self.ratio + 1)) % len(self.cifar_dataset)
                    ]
                else:  # If this index falls within the COCO "slice"
                    return self.coco_dataset[
                        int(index / (self.ratio + 1)) % len(self.coco_dataset)
                    ]

    def __len__(self):
        if self.train:
            return max(len(self.coco_dataset), len(self.cifar_dataset))
        else:
            if self.eval_dataset == "coco":
                return len(self.coco_dataset)
            elif self.eval_dataset == "cifar":
                return len(self.cifar_dataset)
            else:  # Mixed sampling during evaluation
                return max(len(self.coco_dataset), len(self.cifar_dataset))
