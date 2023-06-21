from __future__ import absolute_import, division, print_function

from .tasks.simclr_pretrain import SimCLRPreTrainer
from .tasks.simclr_linear import SimCLRLinearTrainer
from .tasks.mae_pretrain import MAEPreTrainer

train_factory = {
    "simclr_pretrain": SimCLRPreTrainer,
    "simclr_linear": SimCLRLinearTrainer,
    "mae_pretrain": MAEPreTrainer,
}
