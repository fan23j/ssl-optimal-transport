from __future__ import absolute_import, division, print_function

from .tasks.simclr_pretrain import SimCLRPreTrainer
from .tasks.linear import LinearTrainer
from .tasks.mae_pretrain import MAEPreTrainer
from .tasks.classify_anything_multi import ClassifyAnythingMultiTrainer
from .tasks.classify_anything_single import ClassifyAnythingSingleTrainer
from .tasks.swav_pretrain import SwAVPreTrainer
from .tasks.multiclass_linear import MultiClassLinearTrainer

train_factory = {
    "simclr_pretrain": SimCLRPreTrainer,
    "simclr_linear": LinearTrainer,
    "mae_pretrain": MAEPreTrainer,
    "mae_linear": LinearTrainer,
    "classify_anything_multi": ClassifyAnythingMultiTrainer,
    "classify_anything_single": ClassifyAnythingSingleTrainer,
    "swav_pretrain": SwAVPreTrainer,
    "multiclass_linear": MultiClassLinearTrainer,
}
