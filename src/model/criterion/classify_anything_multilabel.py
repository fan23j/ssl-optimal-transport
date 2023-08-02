import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Classify_Anything_MultiLabel_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_MultiLabel_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.weight_decay = cfg.TRAIN.WD

    def forward(self, features, labels_vector, targets, model, **kwargs):
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

        # Compute L2 regularization loss
        l2_reg = 0.0
        for name, param in model.named_parameters():
            if "weight" in name:
                l2_reg += torch.norm(param).item()

        loss = pos_loss + neg_loss + self.weight_decay * l2_reg

        return {"classify_anything_loss": loss}, cosim_matrix
