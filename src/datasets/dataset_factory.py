from .dataset.cifar10 import CIFAR10
from .dataset.cifar100 import CIFAR100
from .dataset.LT_cifar10 import LongTailCIFAR10
from .dataset.LT_cifar100 import LongTailCIFAR100
from .dataset.multi_crop_dataset import MultiCropDataset
from .dataset.LT_imagenet.LT_imagenet import LT_Imagenet
from .dataset.coco import Coco
from .dataset.mixed_dataset_coco_cifar import MixedDatasetCocoCifar
from .dataset.tiny_imagenet import TinyImageNet
from .dataset.nuswide import NUSWIDEClassification
from .dataset.mixed_dataset_coco_imagenet import MixedDatasetCocoImageNet
from .dataset.mixed_dataset_nuswide_cifar import MixedDatasetNuswideCifar
from .dataset.mixed_dataset_nuswide_imagenet import MixedDatasetNuswideImageNet

from .sample.mae import MaeSampler
from .sample.simclr import SimCLRSampler
from .sample.classify_anything import ClassifyAnythingSampler
from .sample.multiclass_inference import MulticlassInferenceSampler


_dataset_factory = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "LT_CIFAR10": LongTailCIFAR10,
    "LT_CIFAR100": LongTailCIFAR100,
    "MULTICROP": MultiCropDataset,
    "LT_IMAGENET": LT_Imagenet,
    "COCO": Coco,
    "MIXED_COCO_CIFAR": MixedDatasetCocoCifar,
    "TINY_IMAGENET": TinyImageNet,
    "NUSWIDE": NUSWIDEClassification,
    "MIXED_COCO_IMAGENET": MixedDatasetCocoImageNet,
    "MIXED_NUSWIDE_CIFAR": MixedDatasetNuswideCifar,
    "MIXED_NUSWIDE_IMAGENET": MixedDatasetNuswideImageNet,
}

_sample_factory = {
    "SIMCLR": SimCLRSampler,
    "MAE": MaeSampler,
    "CLASSIFY_ANYTHING": ClassifyAnythingSampler,
    "MULTICLASS_INFERENCE": MulticlassInferenceSampler,
    "NONE": None,
}


def get_dataset(cfg):
    sampler = _sample_factory[cfg.DATASET.SAMPLE]
    if sampler is not None:
        sampler = sampler(cfg)
    dataset = _dataset_factory[cfg.DATASET.DATASET]

    # (train_dataset, test_dataset)
    return (
        dataset(cfg, cfg.DATA_DIR, train=False, download=True, sampler=sampler),
        dataset(cfg, cfg.DATA_DIR, train=False, download=True, sampler=sampler),
    )
