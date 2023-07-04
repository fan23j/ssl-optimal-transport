from __future__ import absolute_import, division, print_function

from .tasks.simclr_pretrain import SimCLRPreTrainer
from .tasks.simclr_linear import SimCLRLinearTrainer
from .tasks.mae_pretrain import MAEPreTrainer
from .tasks.mae_linear import MAELinearTrainer
from .tasks.simclr_classify_anything import SimCLRClassifyAnythingTrainer
from .tasks.swav_pretrain import SwAVPreTrainer

train_factory = {
    "simclr_pretrain": SimCLRPreTrainer,
    "simclr_linear": SimCLRLinearTrainer,
    "mae_pretrain": MAEPreTrainer,
    "mae_linear": MAELinearTrainer,
    "simclr_classify_anything": SimCLRClassifyAnythingTrainer,
    "swav_pretrain": SwAVPreTrainer,
}
