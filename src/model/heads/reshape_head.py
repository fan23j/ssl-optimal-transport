from __future__ import absolute_import, division, print_function

import torch.nn as nn


class ReshapeHead(nn.Module):
    def __init__(self, cfg):
        super(ReshapeHead, self).__init__()

        self.head = nn.Linear(
            cfg.MODEL.FEATURE_DIM, cfg.MODEL.OUTPUT_FEATURES, bias=True
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        features, out = x
        return self.head(out)
