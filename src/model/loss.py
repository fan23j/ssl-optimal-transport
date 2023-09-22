import torch.nn as nn

from .criterion.infonce import Info_NCE_Loss
from .criterion.sinkhorn import Sinkhorn_Loss
from .criterion.gromov_wasserstein.gromov_wasserstein import Gromov_Wasserstein_Loss
from .criterion.cross_entropy import Cross_Entropy_Loss
from .criterion.mae_mse import Mae_MSE_Loss
from .criterion.swav import SwAV_Loss
from .criterion.classify_anything import Classify_Anything_Loss
from .criterion.classify_anything_multilabel import Classify_Anything_MultiLabel_Loss
from .criterion.classify_anything_mixed import Classify_Anything_Mixed_Loss
from .criterion.classify_anything_mixed_ot import Classify_Anything_Mixed_OT_Loss
from .criterion.bce import BCE_Loss
from .criterion.inference_sinkhorn import Inference_Sinkhorn_Loss
from .criterion.inference_multilabel import Inference_Multilabel_Loss
from .criterion.classify_anything_mixed_ot_sinkhorn import Classify_Anything_Mixed_OT_Sinkhorn_Loss

_loss_factory = {
    "infonce": Info_NCE_Loss,
    "sinkhorn": Sinkhorn_Loss,
    "gromov_wasserstein": Gromov_Wasserstein_Loss,
    "cross_entropy": Cross_Entropy_Loss,
    "mae_mse": Mae_MSE_Loss,
    "swav": SwAV_Loss,
    "classify_anything": Classify_Anything_Loss,
    "classify_anything_multi": Classify_Anything_MultiLabel_Loss,
    "classify_anything_mixed": Classify_Anything_Mixed_Loss,
    "classify_anything_mixed_ot": Classify_Anything_Mixed_OT_Loss,
    "bce": BCE_Loss,
    "inference_sinkhorn": Inference_Sinkhorn_Loss,
    "inference_multilabel": Inference_Multilabel_Loss,
    "classify_anything_mixed_ot_sinkhorn": Classify_Anything_Mixed_OT_Sinkhorn_Loss,
}


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.losses = [_loss_factory[name](cfg=cfg) for name in cfg.LOSS.METRIC]

    def forward(self, **kwargs):
        total_loss = 0
        loss_states = {}
        loss_aux = None
        for loss in self.losses:
            loss_output = loss(**kwargs)
            if isinstance(loss_output, tuple):
                loss_dict, loss_aux = loss_output
            else:
                loss_dict = loss_output
            total_loss += list(loss_dict.values())[0]
            loss_states.update(loss_dict)
        loss_states["loss"] = total_loss
        return total_loss, loss_states, loss_aux
