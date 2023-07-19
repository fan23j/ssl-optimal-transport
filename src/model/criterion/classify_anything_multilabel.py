import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Classify_Anything_MultiLabel_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_MultiLabel_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE

    def forward(self, features, labels_vector, targets, **kwargs):
        """
        features: [B, 300]
        label_vector: [num_class, 300]
        targets: [B, num_class]
        """

        cosim_matrix = torch.matmul(features, labels_vector.t()) / self.temperature

        cosim_matrix_sigmoid = torch.sigmoid(cosim_matrix)

        pos_mask = targets.bool()
        neg_mask = ~pos_mask

        pos_loss = -torch.log(cosim_matrix_sigmoid[pos_mask]).mean()
        neg_loss = -torch.log(1 - cosim_matrix_sigmoid[neg_mask]).mean()

        loss = pos_loss + neg_loss

        return {"classify_anything_loss": loss}, cosim_matrix
