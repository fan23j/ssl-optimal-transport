import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random

from PIL import ImageFilter


class MultiCropDataset:
    def __init__(
        self,
        cfg,
        root,
        train=True,
        download=False,
        return_index=False,
        sampler=None,
    ):
        self.dataset_class = get_dataset_class(cfg.DATASET.DATASET)
        self.dataset = self.dataset_class(root, train=train, download=download)
        self.sampler = sampler

        self.size_crops = cfg.DATASET.MULTICROP_SIZE_CROPS
        self.nmb_crops = cfg.DATASET.MULTICROP_NMB_CROPS
        self.min_scale_crops = cfg.DATASET.MULTICROP_MIN_SCALE_CROPS
        self.max_scale_crops = cfg.DATASET.MULTICROP_MAX_SCALE_CROPS

        assert len(self.size_crops) == len(self.nmb_crops)
        assert len(self.min_scale_crops) == len(self.nmb_crops)
        assert len(self.max_scale_crops) == len(self.nmb_crops)
        self.return_index = return_index

        color_transform = [
            get_color_distortion(s=cfg.DATASET.COLOR_JITTER_STRENGTH),
            PILRandomGaussianBlur(),
        ]
        mean = cfg.DATASET.MEAN
        std = cfg.DATASET.STD
        trans = []
        for i in range(len(self.size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )
            trans.extend(
                [
                    transforms.Compose(
                        [
                            randomresizedcrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
                * self.nmb_crops[i]
            )
        self.trans = trans

    def __getitem__(self, index):
        if isinstance(self.dataset, datasets.ImageFolder):
            # Use path-based image loading for ImageFolder
            path, _ = self.dataset.samples[index]
            image = self.dataset.loader(path)
        else:
            # Use in-memory image loading for CIFAR datasets
            image, _ = self.dataset[index]

        multi_crops = list(map(lambda trans: trans(image), self.trans))

        if self.return_index:
            return index, multi_crops

        return multi_crops

    def __len__(self):
        return len(self.dataset)


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def get_dataset_class(dataset_name):
    if dataset_name.lower() == "CIFAR10":
        return datasets.CIFAR10
    elif dataset_name.lower() == "CIFAR100":
        return datasets.CIFAR100
    elif dataset_name.lower() == "IMAGENET":
        return datasets.ImageFolder
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
