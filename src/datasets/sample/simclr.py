from PIL import Image
from .sampler import Sampler


class SimCLRSampler(Sampler):
    """Sampler for SimCLR."""

    def sample(dataset, img, target):
        img = Image.fromarray(img)

        if dataset.transform is not None:
            out_1 = dataset.transform(img)
            out_2 = dataset.transform(img)

        if dataset.target_transform is not None:
            target = dataset.target_transform(target)

        return {"out_1": out_1, "out_2": out_2, "target": target}
