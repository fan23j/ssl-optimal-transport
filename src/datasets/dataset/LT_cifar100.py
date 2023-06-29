import numpy as np

from .cifar100 import CIFAR100


class LongTailCIFAR100(CIFAR100):
    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        super(LongTailCIFAR100, self).__init__(
            cfg, root, train=train, download=download, sampler=sampler
        )

        self.num_classes = 100

        self.imbalance_ratio = cfg.DATASET.LT_IMBALANCE_RATIO

        # Define your long-tail distribution based on the imbalance ratio
        max_samples = 600  # Maximum number of samples in a class in CIFAR-100
        min_samples = int(
            max_samples / self.imbalance_ratio
        )  # Minimum number of samples in a class

        num_samples_per_class = np.linspace(
            min_samples, max_samples, num=self.num_classes
        ).astype(int)

        targets = np.array(self.targets)
        balanced_idx = []

        for i in range(self.num_classes):
            idx = np.where(targets == i)[0][: num_samples_per_class[i]]
            balanced_idx.extend(idx)

        np.random.shuffle(balanced_idx)
        self.data = self.data[balanced_idx]
        self.targets = list(targets[balanced_idx])
