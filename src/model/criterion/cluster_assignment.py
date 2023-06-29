import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Cluster_Assignment_Loss(nn.Module):
    def __init__(self, cfg):
        super(Cluster_Assignment_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.nmb_crops = cfg.DATASET.MULTICROP_NMB_CROPS

    def forward(self, output, q, crop_id):
        """
        Calculate Cross-Entropy loss of cluster assignments.

        Args:
            q: assignments from sinkhorn.

        Returns:
            Cluster assignment loss.
        """
        subloss = 0
        for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
            x = (
                output[self.batch_size * v : self.batch_size * (v + 1)]
                / self.temperature
            )
            subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

        return {"cluster_assignment_loss": subloss}
