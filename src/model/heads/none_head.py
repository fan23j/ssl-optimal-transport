from __future__ import absolute_import, division, print_function

import torch.nn as nn


class NoneHead(nn.Module):
    def __init__(self, cfg):
        super(NoneHead, self).__init__()

    def forward(self, x):
        return x
