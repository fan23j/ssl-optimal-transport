from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, cfg):
        super(ProjectionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(2048, cfg.MODEL.HIDDEN_MLP, bias=False),
            nn.BatchNorm1d(cfg.MODEL.HIDDEN_MLP),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.MODEL.HIDDEN_MLP, cfg.MODEL.FEATURE_DIM, bias=True),
        )

        if cfg.MODEL.NMB_PROTOTYPES > 0:
            self.prototypes = nn.Linear(
                cfg.MODEL.FEATURE_DIM, cfg.MODEL.NMB_PROTOTYPES, bias=False
            )

        self.init_weights()

    def init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head.apply(init_func)

    def forward(self, x):
        out = self.head(x)
        if self.prototypes is not None:
            return F.normalize(out, dim=-1), self.prototypes(out)
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1)
