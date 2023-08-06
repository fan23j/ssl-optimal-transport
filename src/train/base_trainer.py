from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from model.loss import Loss


class BaseTrainer(object):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        lr_scheduler=None,
        train_dataset=None,
        val_dataset=None,
    ):
        self.cfg = cfg
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = self._get_losses(cfg)
        self.model = model
        self.temperature = cfg.LOSS.TEMPERATURE
        self.local_rank = len(cfg.GPUS) - 1
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset

    def set_device(self, device):
        if self.cfg.TRAIN.DISTRIBUTE:
            self.model = self.model.to(device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=True,
                device_ids=[
                    self.local_rank,
                ],
                output_device=self.local_rank,
            )
        else:
            self.model = nn.DataParallel(self.model).to(device)
        self.loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def _get_losses(self, cfg):
        return Loss(cfg)

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def train(self, epoch, data_loader):
        raise NotImplementedError

    def val(self, epoch, data_loader, val_dataset=None):
        raise NotImplementedError

    def test(self, data_loader, test_dataset=None):
        raise NotImplementedError
