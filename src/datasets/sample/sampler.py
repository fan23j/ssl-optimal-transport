from abc import ABC, abstractmethod


class Sampler(ABC):
    """Abstract class for samplers."""

    @abstractmethod
    def sample(self, dataset, img, target):
        pass
