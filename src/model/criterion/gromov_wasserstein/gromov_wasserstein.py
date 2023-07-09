import torch
import torch.nn as nn
import torch.nn.functional as F

from .gw import entropic_gromov_wasserstein


class Gromov_Wasserstein_Loss(nn.Module):
    def __init__(self, cfg):
        super(Gromov_Wasserstein_Loss, self).__init__()
        self.temperature = cfg.LOSS.TEMPERATURE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.world_size = len(cfg.GPUS) if cfg.TRAIN.DISTRIBUTE else 1
        self.gw_max_iter = cfg.LOSS.GW_MAX_ITER
        self.gw_epsilon = cfg.LOSS.GW_EPSILON
        self.sinkhorn_max_iter = cfg.LOSS.SINKHORN_MAX_ITER
        self.gamma = cfg.LOSS.GW_GAMMA

    def forward(self, features_1, features_2, **kwargs):
        out_1 = F.normalize(features_1, dim=1)
        out_2 = F.normalize(features_2, dim=1)

        # compute distance matrix
        daa = 1.0 - torch.matmul(out_1, out_1.t())
        dbb = 1.0 - torch.matmul(out_2, out_2.t())

        # normalize distance matrix
        daa = daa / daa.max()
        dbb = dbb / dbb.max()

        n = daa.shape[0]

        # init p and q
        p = torch.full((n,), 1.0 / n, device=daa.device)
        q = torch.full((n,), 1.0 / n, device=dbb.device)

        gw = entropic_gromov_wasserstein(
            daa,
            dbb,
            p,
            q,
            "square_loss",
            epsilon=self.gw_epsilon,
            max_iter=self.gw_max_iter,
            sinkhorn_max_iter=self.sinkhorn_max_iter,
            verbose=True,
        )

        # small epsilon to avoid nan
        epsilon = 1e-10

        # compute loss
        loss = -torch.log(torch.diag(gw) + epsilon).sum()

        return {"gw_loss": loss * self.gamma}
