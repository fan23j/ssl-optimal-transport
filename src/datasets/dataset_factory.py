import torchvision
import torchvision.transforms as transforms
from .transforms import Transforms

from .dataset.cifar10 import CIFAR10
from .dataset.cifar100 import CIFAR100


def get_dataset(cfg):
    transforms = Transforms(cfg)
    train_transform, test_transform = transforms()
    if cfg.DATASET.DATASET == "STL10":
        train_dataset = torchvision.datasets.STL10(
            cfg.DATA_DIR,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            cfg.DATA_DIR,
            split="test",
            download=True,
            transform=test_transform,
        )
    elif cfg.DATASET.DATASET == "CIFAR10":
        train_dataset = CIFAR10(
            cfg.DATA_DIR,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = CIFAR10(
            cfg.DATA_DIR,
            train=False,
            download=True,
            transform=test_transform,
        )
    elif cfg.DATASET.DATASET == "CIFAR100":
        train_dataset = CIFAR100(
            cfg.DATA_DIR,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = CIFAR100(
            cfg.DATA_DIR,
            train=False,
            download=True,
            transform=test_transform,
        )
    elif cfg.DATASET.DATASET == "SVHN":
        train_dataset = torchvision.datasets.SVHN(
            cfg.DATA_DIR,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.SVHN(
            cfg.DATA_DIR,
            split="test",
            download=True,
            transform=test_transform,
        )
    elif cfg.DATASET.DATASET == "ImageNet100":
        train_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet100/train",
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet100/val",
            transform=test_transform,
        )
    elif cfg.DATASET.DATASET == "ImageNet1k":
        train_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet/train",
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet/val",
            transform=test_transform,
        )
    else:
        raise NotImplementedError

    return train_dataset, test_dataset
