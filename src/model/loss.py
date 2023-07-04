import torch.nn as nn

from .criterion.infonce import Info_NCE_Loss
from .criterion.sinkhorn import Sinkhorn_Loss
from .criterion.gromov_wasserstein.gromov_wasserstein import Gromov_Wasserstein_Loss
from .criterion.cross_entropy import Cross_Entropy_Loss
from .criterion.mae_mse import Mae_MSE_Loss
from .criterion.swav import SwAV_Loss

_loss_factory = {
    "infonce": Info_NCE_Loss,
    "sinkhorn": Sinkhorn_Loss,
    "gromov_wasserstein": Gromov_Wasserstein_Loss,
    "cross_entropy": Cross_Entropy_Loss,
    "mae_mse": Mae_MSE_Loss,
    "swav": SwAV_Loss,
}


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.losses = [_loss_factory[name](cfg=cfg) for name in cfg.LOSS.METRIC]

    def forward(self, *args):
        total_loss = 0
        loss_states = {}
        for loss in self.losses:
            loss_dict = loss(*args)
            total_loss += list(loss_dict.values())[0]
            loss_states.update(loss_dict)
        loss_states["loss"] = total_loss
        return total_loss, loss_states
