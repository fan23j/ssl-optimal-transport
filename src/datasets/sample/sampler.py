from abc import ABC, abstractmethod
import torch


class Sampler(ABC):
    """Abstract class for samplers."""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.TASK == "classify_anything_mixed":
            self.label_vectors = torch.load(cfg.MODEL.LABEL_VECTORS)

    @abstractmethod
    def sample(self, dataset, img, target):
        pass
