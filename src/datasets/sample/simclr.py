from PIL import Image
from .sampler import Sampler


class SimCLRSampler(Sampler):
    """Sampler for SimCLR."""

    def sample(self, dataset, img, target):
        try:
            img = Image.fromarray(img)
        except AttributeError:
            pass

        if dataset.transform is not None:
            out_1 = dataset.transform(img)
            out_2 = dataset.transform(img)

        out_3 = dataset.test_transform(img)

        if dataset.target_transform is not None:
            target = dataset.target_transform(target)

        return {"out_1": out_1, "out_2": out_2, "target": target, "out_3": out_3}
