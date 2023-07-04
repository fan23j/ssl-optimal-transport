import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Classify_Anything_Loss(nn.Module):
    def __init__(self, cfg):
        super(Classify_Anything_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.sinkhorn_iters = cfg.LOSS.SINKHORN_MAX_ITER

    def sinkhorn_knopp(self, cost_matrix, epsilon, num_iters):
        """
        Calculate the Sinkhorn normalization.

        Args:
            cost_matrix: cost matrix between all samples in the batch
            epsilon: scaling factor for normalization
            num_iters: number of iterations to run sinkhorn-knopp

        Returns:
            p_matrix: the final row/column normalized matrix
        """
        # Compute the initial p_matrix
        p_matrix = torch.exp(-cost_matrix / epsilon)

        # Perform Sinkhorn iterations
        for i in range(num_iters):
            # Transpose every other iteration
            if i != 0:
                p_matrix = p_matrix.t()

            # Row normalize
            p_matrix = p_matrix / p_matrix.sum(dim=1, keepdim=True)

        # If num_iters is even, transpose p_matrix one more time to bring it back to original orientation
        if num_iters % 2 == 0:
            p_matrix = p_matrix.t()

        return p_matrix

    def forward(self, features, labels_vector):
        """
        features: [B, 300]
        label_vector: [B, 300]
        """
        sim_matrix = torch.exp(
            torch.matmul(features, labels_vector.t().contiguous()) / self.temperature
        )

        cost_matrix = 1.0 - sim_matrix
        cost_matrix.fill_diagonal_(float("inf"))
        p_matrix = self.sinkhorn_knopp(
            cost_matrix, self.temperature, self.sinkhorn_iters
        )

        prob_i = torch.diag(p_matrix, self.batch_size)
        prob_j = torch.diag(p_matrix, -self.batch_size)

        # Concatenate along the first dimension.
        prob_p = torch.cat([prob_i, prob_j], dim=0)

        # Calculate the loss.
        loss = -torch.log(prob_p).sum() / (2 * self.batch_size)

        return {"classify_anything_loss": loss}
