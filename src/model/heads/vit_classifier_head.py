import torch.nn as nn
import torch


class ViTClassifierHead(nn.Module):
    def __init__(self, cfg):
        super(ViTClassifierHead, self).__init__()
        self.linear = torch.nn.Linear(
            cfg.MODEL.MAE_EMBED_DIM,
            cfg.DATASET.NUM_CLASSES,
        )
        self.initialize_weights()

    def forward(self, x):
        features, _ = x
        logits = self.linear(features[0])
        return logits

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)
