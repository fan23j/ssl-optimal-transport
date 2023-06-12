import torchvision
from .transforms import TransformsSimCLR

def get_dataset(cfg, train=True):
    transform = (
        TransformsSimCLR(size=cfg.DATASET.IMAGE_SIZE) if train else TransformsSimCLR(size=cfg.DATASET.IMAGE_SIZE).test_transform
    )
    if cfg.DATASET.DATASET == "STL10":
        train_dataset = torchvision.datasets.STL10(
            cfg.DATA_DIR,
            split="train",
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.STL10(
            cfg.DATA_DIR,
            split="test",
            download=True,
            transform=transform,
        )
    elif cfg.DATASET.DATASET == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            cfg.DATA_DIR,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            cfg.DATA_DIR,
            train=False,
            download=True,
            transform=transform,
        )
    elif cfg.DATASET.DATASET == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            cfg.DATA_DIR,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            cfg.DATA_DIR,
            train=False,
            download=True,
            transform=transform,
        )
    elif cfg.DATASET.DATASET == "SVHN":
        train_dataset = torchvision.datasets.SVHN(
            cfg.DATA_DIR,
            split="train",
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.SVHN(
            cfg.DATA_DIR,
            split="test",
            download=True,
            transform=transform,
        )
    elif cfg.DATASET.DATASET == "ImageNet100":
        train_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet100/train",
            transform=transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet100/val",
            transform=transform,
        )
    elif cfg.DATASET.DATASET == "ImageNet1k":
        train_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet/train",
            transform=transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            "/mnt/nas/dataset_share/imagenet/val",
            transform=transform,
        )
    else:
        raise NotImplementedError

    return train_dataset, test_dataset