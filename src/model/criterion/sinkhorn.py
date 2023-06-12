import torch
import torch.nn as nn
import torch.nn.functional as F

class Sinkhorn_Loss(nn.Module):
    def __init__(self, cfg):
        super(Sinkhorn_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.world_size = len(cfg.GPUS) if cfg.TRAIN.DISTRIBUTE else 1
        self.sinkhorn_iters = cfg.LOSS.SINKHORN_ITERS
        self.gamma = cfg.LOSS.SINKHORN_GAMMA

    def cosine_similarity(self, out_1, out_2):
        """
        Construct the cosine similarity matrix.

        Args:
            out_1, out_2: The output embeddings from the projection head for the two positive pairs.

        Returns:
            The cosine similarity matrix.
        """
        # Concatenate the output embeddings from the projection head
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)

        # Normalizing the output embeddings
        out = F.normalize(out, dim=1)

        # Calculate cosine similarity
        sim_matrix = torch.matmul(out, out.t())
        
        return sim_matrix

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
    
    def forward(self, out_1, out_2):
        sim_matrix = self.cosine_similarity(out_1, out_2)
        cost_matrix = 1.0 - sim_matrix
        cost_matrix.fill_diagonal_(float('inf'))
        p_matrix = self.sinkhorn_knopp(cost_matrix, self.temperature, self.sinkhorn_iters)

        prob_i = torch.diag(p_matrix, self.batch_size)
        prob_j = torch.diag(p_matrix, -self.batch_size)

        # Concatenate along the first dimension.
        prob_p = torch.cat([prob_i, prob_j], dim=0)

        # Calculate the loss.
        loss = -torch.log(prob_p).sum() / (2 * self.batch_size * self.world_size)

        return {'sinkhorn_loss': loss * self.gamma}