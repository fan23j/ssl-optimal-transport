import torch
import math


class OptimizerSchedulerFactory:
    def __init__(self, cfg, model):
        self.model = model
        self.cfg = cfg

    def create(self):
        if self.cfg.TRAIN.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.TRAIN.LR,
                weight_decay=self.cfg.TRAIN.WD,
            )
            lr_func = lambda epoch: 1
        elif self.cfg.TRAIN.OPTIMIZER == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.TRAIN.LR * self.cfg.TRAIN.BATCH_SIZE / 256,
                betas=(0.9, 0.95),
                weight_decay=self.cfg.TRAIN.WD,
            )
            lr_func = lambda epoch: min(
                (epoch + 1) / (self.cfg.TRAIN.WARMUP_EPOCHS + 1e-8),
                0.5 * (math.cos(epoch / self.cfg.TRAIN.EPOCHS * math.pi) + 1),
            )
        elif self.cfg.TRAIN.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.TRAIN.LR,
                momentum=self.cfg.TRAIN.MOMENTUM,
            )
            lr_func = lambda epoch: 1
        else:
            raise NotImplementedError("Optimizer not supported")

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_func, verbose=True
        )

        return optimizer, lr_scheduler
