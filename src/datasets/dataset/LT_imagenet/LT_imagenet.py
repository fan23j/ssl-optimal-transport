import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LT_Imagenet(Dataset):
    """IMAGENET_LT Dataset."""

    def __init__(self, cfg, root, train=True, download=False, sampler=None):
        self.img_path = []
        self.labels = []
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.DATASET.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    cfg.DATASET.COLOR_JITTER[0],
                    cfg.DATASET.COLOR_JITTER[1],
                    cfg.DATASET.COLOR_JITTER[2],
                    cfg.DATASET.COLOR_JITTER[3],
                ),
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.DATASET.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )

        self.transform = self.train_transform if train else self.test_transform

        with open("ImageNet_LT_train.txt" if train else "ImageNet_LT_val.txt") as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index
