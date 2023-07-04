import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Classify_Anything_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE

    def forward(self, features, label_vector):
        """
        features: [B, 128]
        label_vector: [B, 300]
        """
        loss = 0

        return {"classify_anything_loss": loss}
