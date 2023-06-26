from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms


class CIFAR100(CIFAR100):
    """CIFAR100 Dataset."""

    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        super().__init__(root, train=train, download=download)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.DATASET.RANDOM_RESIZED_CROP),
                transforms.RandomHorizontalFlip(p=cfg.DATASET.RANDOM_HORIZONTAL_FLIP),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            cfg.DATASET.COLOR_JITTER[0],
                            cfg.DATASET.COLOR_JITTER[1],
                            cfg.DATASET.COLOR_JITTER[2],
                            cfg.DATASET.COLOR_JITTER[3],
                        )
                    ],
                    p=cfg.DATASET.COLOR_JITTER[4],
                ),
                transforms.RandomGrayscale(p=cfg.DATASET.RANDOM_GRAYSCALE),
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )

        self.transform = self.train_transform if train else self.test_transform

        self.sampler = sampler

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.sampler is not None:
            return self.sampler.sample(self, img, target)

        return img, target
