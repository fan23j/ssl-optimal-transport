from __future__ import absolute_import, division, print_function

from .tasks.simclr_pretrain import SimCLRPreTrainer

train_factory = {
    'simclr_pretrain': SimCLRPreTrainer,
}