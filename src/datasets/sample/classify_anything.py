from PIL import Image
from .sampler import Sampler


class ClassifyAnythingSampler(Sampler):
    """Sampler for ClassifyAnything."""

    def sample(dataset, img, target):
        img = Image.fromarray(img)

        if dataset.transform is not None:
            out = dataset.transform(img)

        if dataset.target_transform is not None:
            target = dataset.target_transform(target)

        return {"out_1": out, "target": target, "label": dataset.class_labels[target]}
