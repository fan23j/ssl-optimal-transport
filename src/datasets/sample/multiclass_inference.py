from torchvision.transforms import functional as TF
from torch.nn.functional import one_hot
from PIL import Image
from .sampler import Sampler
import torch

class MulticlassInferenceSampler(Sampler):

    def sample(self, dataset, img, target):
        try:
            img = Image.fromarray(img)
        except AttributeError:
            pass

        if dataset.transform is not None:
            out_1 = dataset.transform(img)
        
        num_classes = dataset.cls_num
        target_tensor = torch.tensor(target, dtype=torch.int64)
        target_one_hot = one_hot(target_tensor, num_classes)

        return {"out_1": out_1, "target": target_one_hot}
