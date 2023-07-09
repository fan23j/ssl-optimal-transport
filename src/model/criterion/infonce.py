import torch
import torch.nn as nn


class Info_NCE_Loss(nn.Module):
    def __init__(self, cfg):
        super(Info_NCE_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.world_size = len(cfg.GPUS) if cfg.TRAIN.DISTRIBUTE else 1
        self.gamma = cfg.LOSS.INFONCE_GAMMA

    def mask_correlated_samples(self, sim_matrix):
        """
        Mask the self-similarity entries from the similarity matrix.
        """
        # Create a boolean mask for the samples that should be used for contrastive loss
        mask = (
            torch.ones_like(sim_matrix)
            - torch.eye(2 * self.batch_size, device=sim_matrix.device)
        ).bool()

        # Apply the mask to the similarity matrix
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        return sim_matrix

    def forward(self, out_1, out_2, **kwargs):
        # Concatenate the output embeddings from the projection head
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)

        # Compute similarity matrix between all samples in the batch
        # [2*B, 2*B]
        sim_matrix = torch.exp(
            torch.matmul(out, out.t().contiguous()) / self.temperature
        )

        # Mask self-similarities
        sim_matrix = self.mask_correlated_samples(sim_matrix)

        # Compute positive pair similarities
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)

        # Duplicate positive pair similarities to match size of the similarity matrix
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        # Compute the InfoNCE loss
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return {"infonce_loss": loss * self.gamma}
