import torch
from PIL import Image
from .sampler import Sampler


class ClassifyAnythingSampler(Sampler):
    """Sampler for ClassifyAnything."""

    def sample(self, dataset, img, target):
        try:
            img = Image.fromarray(img)
        except TypeError:
            pass

        if dataset.transform is not None:
            out = dataset.transform(img)

        if isinstance(target, torch.Tensor):
            # If target is a one-hot tensor, get the labels of the "1" entries
            target_labels = [
                dataset.class_labels[i] for i, x in enumerate(target) if x == 1
            ]
        else:
            # If target is already an index, just use it directly to get the label
            target_labels = [dataset.class_labels[target]]

        # Create a list of indices from target_labels based on self.label_vectors keys
        target_indices = [
            list(self.label_vectors.keys()).index(label) for label in target_labels
        ]

        # Initialize a tensor for one-hot vectors
        targets = torch.zeros(len(self.label_vectors.keys()))

        # Populate one-hot vectors using target indices
        targets[target_indices] = 1

        return {"out_1": out, "target": targets}
