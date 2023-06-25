import torch.nn as nn
import torch


class ViTClassifierHead(nn.Module):
    def __init__(self, cfg):
        super(ViTClassifierHead, self).__init__()
        self.linear = torch.nn.Linear(
            cfg.MODEL.MAE_EMBED_DIM,
            cfg.DATASET.NUM_CLASSES,
        )

    def forward(self, x):
        features, _ = x
        logits = self.linear(features[0])
        return logits
