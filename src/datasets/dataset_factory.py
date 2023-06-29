from .dataset.cifar10 import CIFAR10
from .dataset.cifar100 import CIFAR100
from .dataset.LT_cifar10 import LongTailCIFAR10
from .dataset.LT_cifar100 import LongTailCIFAR100
from .dataset.multi_crop_dataset import MultiCropDataset

from .sample.mae import MaeSampler
from .sample.simclr import SimCLRSampler


_dataset_factory = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "LT_CIFAR10": LongTailCIFAR10,
    "LT_CIFAR100": LongTailCIFAR100,
    "MULTICROP": MultiCropDataset,
}

_sample_factory = {
    "SIMCLR": SimCLRSampler,
    "MAE": MaeSampler,
    "NONE": None,
}


def get_dataset(cfg):
    sampler = _sample_factory[cfg.DATASET.SAMPLE]
    dataset = _dataset_factory[cfg.DATASET.DATASET]

    # (train_dataset, test_dataset)
    return (
        dataset(cfg, cfg.DATA_DIR, train=True, download=True, sampler=sampler),
        dataset(cfg, cfg.DATA_DIR, train=False, download=True, sampler=sampler),
    )


# def get_dataset(cfg):
#     transforms = Transforms(cfg)
#     train_transform, test_transform = transforms()
#     if cfg.DATASET.DATASET == "STL10":
#         train_dataset = torchvision.datasets.STL10(
#             cfg.DATA_DIR,
#             split="train",
#             download=True,
#             transform=train_transform,
#         )
#         test_dataset = torchvision.datasets.STL10(
#             cfg.DATA_DIR,
#             split="test",
#             download=True,
#             transform=test_transform,
#         )
#     elif cfg.DATASET.DATASET == "CIFAR10":
#         train_dataset = CIFAR10(
#             cfg.DATA_DIR,
#             train=True,
#             download=True,
#             transform=train_transform,
#         )
#         test_dataset = CIFAR10(
#             cfg.DATA_DIR,
#             train=False,
#             download=True,
#             transform=test_transform,
#         )
#     elif cfg.DATASET.DATASET == "CIFAR100":
#         train_dataset = CIFAR100(
#             cfg.DATA_DIR,
#             train=True,
#             download=True,
#             transform=train_transform,
#         )
#         test_dataset = CIFAR100(
#             cfg.DATA_DIR,
#             train=False,
#             download=True,
#             transform=test_transform,
#         )
#     elif cfg.DATASET.DATASET == "SVHN":
#         train_dataset = torchvision.datasets.SVHN(
#             cfg.DATA_DIR,
#             split="train",
#             download=True,
#             transform=train_transform,
#         )
#         test_dataset = torchvision.datasets.SVHN(
#             cfg.DATA_DIR,
#             split="test",
#             download=True,
#             transform=test_transform,
#         )
#     elif cfg.DATASET.DATASET == "ImageNet100":
#         train_dataset = torchvision.datasets.ImageFolder(
#             "/mnt/nas/dataset_share/imagenet100/train",
#             transform=train_transform,
#         )
#         test_dataset = torchvision.datasets.ImageFolder(
#             "/mnt/nas/dataset_share/imagenet100/val",
#             transform=test_transform,
#         )
#     elif cfg.DATASET.DATASET == "ImageNet1k":
#         train_dataset = torchvision.datasets.ImageFolder(
#             "/mnt/nas/dataset_share/imagenet/train",
#             transform=train_transform,
#         )
#         test_dataset = torchvision.datasets.ImageFolder(
#             "/mnt/nas/dataset_share/imagenet/val",
#             transform=test_transform,
#         )
#     else:
#         raise NotImplementedError

#     return train_dataset, test_dataset
