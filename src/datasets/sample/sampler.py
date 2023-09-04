from abc import ABC, abstractmethod
import torch
import json

class Sampler(ABC):
    """Abstract class for samplers."""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.DATASET.MIXED_LABELS:
            with open(cfg.DATASET.MIXED_LABELS, "r") as f:
                self.mixed_labels = json.load(f)

    @abstractmethod
    def sample(self, dataset, img, target):
        pass
