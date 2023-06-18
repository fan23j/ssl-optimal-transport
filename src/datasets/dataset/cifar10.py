from PIL import Image
from torchvision.datasets import CIFAR10


class CIFAR10(CIFAR10):
    """CIFAR10 Dataset."""

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            out_1 = self.transform(img)
            out_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"out_1": out_1, "out_2": out_2, "target": target}
