import torch
import torch.nn as nn


class Classify_Anything_Mixed_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_Mixed_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE


def forward(self, features, label_vectors, targets, **kwargs):
    """
    features: [B, 300]
    label_vectors: [num_class, 300]
    targets: [B, num_class]
    """

    cosim_matrix = torch.matmul(features, label_vectors.t()) / self.temperature

    cosim_matrix_sigmoid = torch.sigmoid(cosim_matrix)

    pos_mask = targets.bool()
    neg_mask = ~pos_mask

    pos_loss = -torch.log(cosim_matrix_sigmoid[pos_mask]).mean()
    neg_loss = -torch.log(1 - cosim_matrix_sigmoid[neg_mask]).mean()

    loss = pos_loss + neg_loss

    return {"classify_anything_loss": loss}, cosim_matrix
