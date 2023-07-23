from __future__ import absolute_import, division, print_function

import torch.nn as nn


class MultiClassifierHead(nn.Module):
    def __init__(self, cfg):
        super(MultiClassifierHead, self).__init__()

        self.head = nn.Sequenntial(
            nn.Linear(cfg.HEAD_INPUT_DIM, cfg.DATASET.NUM_CLASSES, bias=True),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        return self.head(x)
