import torchvision.transforms as transforms


class Transforms:
    def __init__(self, cfg):
        self.cfg = cfg
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

    def __call__(self):
        if self.cfg.DATASET.USE_TEST_AUG:
            return self.test_transform, self.test_transform
        return self.train_transform, self.test_transform
