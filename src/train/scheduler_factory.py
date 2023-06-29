import torch
import math
import numpy as np

from apex.parallel.LARC import LARC


class OptimizerSchedulerFactory:
    def __init__(self, cfg, model, train_loader):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader

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
                weight_decay=self.cfg.TRAIN.WD,
            )
            lr_func = lambda epoch: 1
        elif self.cfg.TRAIN.OPTIMIZER == "larc":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.TRAIN.LR,
                momentum=self.cfg.TRAIN.MOMENTUM,
                weight_decay=self.cfg.TRAIN.WD,
            )
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

            # Create the learning rate schedules for warmup and cosine annealing
            warmup_lr_schedule = np.linspace(
                self.cfg.TRAIN.START_WARMUP,
                self.cfg.TRAIN.BASE_LR,
                len(self.train_loader) * self.cfg.TRAIN.WARMUP_EPOCHS,
            )
            iters = np.arange(
                len(self.train_loader)
                * (self.cfg.TRAIN.EPOCHS - self.cfg.TRAIN.WARMUP_EPOCHS)
            )
            cosine_lr_schedule = np.array(
                [
                    self.cfg.TRAIN.FINAL_LR
                    + 0.5
                    * (self.cfg.TRAIN.BASE_LR - self.cfg.TRAIN.FINAL_LR)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * t
                            / (
                                len(self.train_loader)
                                * (self.cfg.TRAIN.EPOCHS - self.cfg.TRAIN.WARMUP_EPOCHS)
                            )
                        )
                    )
                    for t in iters
                ]
            )
            lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

            lr_func = lambda epoch: lr_schedule[epoch]
        else:
            raise NotImplementedError("Optimizer not supported")

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_func, verbose=True
        )

        return optimizer, lr_scheduler
