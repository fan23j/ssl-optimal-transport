import numpy as np

from .cifar10 import CIFAR10


class LongTailCIFAR10(CIFAR10):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        super(LongTailCIFAR10, self).__init__(
            cfg, root, train=train, download=download, sampler=sampler
        )

        self.num_classes = 10

        self.imbalance_ratio = cfg.DATASET.LT_IMBALANCE_RATIO

        # Define your long-tail distribution based on the imbalance ratio
        max_samples = 6000  # Maximum number of samples in a class
        min_samples = int(
            max_samples / self.imbalance_ratio
        )  # Minimum number of samples in a class

        num_samples_per_class = np.linspace(
            min_samples, max_samples, num=self.num_classes
        ).astype(int)

        # This part actually selects instances according to your distribution
        targets = np.array(self.targets)
        balanced_idx = []

        for i in range(self.num_classes):
            idx = np.where(targets == i)[0][: num_samples_per_class[i]]
            balanced_idx.extend(idx)

        np.random.shuffle(balanced_idx)
        self.data = self.data[balanced_idx]
        self.targets = list(targets[balanced_idx])
