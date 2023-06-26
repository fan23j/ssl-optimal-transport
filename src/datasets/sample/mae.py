from PIL import Image
from .sampler import Sampler


class MaeSampler(Sampler):
    """Sampler for MAE."""

    def sample(dataset, img, target):
        img = Image.fromarray(img)

        if dataset.test_transform is not None:
            out = dataset.test_transform(img)

        if dataset.target_transform is not None:
            target = dataset.target_transform(target)

        return {"out_1": out, "target": target}
