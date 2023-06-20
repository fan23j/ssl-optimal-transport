from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, cfg):
        super(ProjectionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.MODEL.FEATURE_DIM, bias=True),
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
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1)
