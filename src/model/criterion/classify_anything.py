import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Classify_Anything_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE

    def forward(self, features, text_features, targets, **kwargs):
        """
        features: [B, 300]
        text_features: [num_class, 512]
        targets: [B]
        """
        # Normalize features and labels_vector along the feature dimension
        # features_norm = F.normalize(features, dim=1)
        # labels_vector_norm = F.normalize(labels_vector, dim=1)

        cosim_matrix = torch.matmul(features, text_features.t()) / self.temperature

        cosim_softmax = F.softmax(cosim_matrix, dim=1)

        loss = F.cross_entropy(cosim_matrix, targets)

        return {"classify_anything_loss": loss}, cosim_softmax
