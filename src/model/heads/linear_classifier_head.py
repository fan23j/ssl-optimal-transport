from __future__ import absolute_import, division, print_function

import torch.nn as nn


class LinearClassifierHead(nn.Module):

    def __init__(self, cfg):
        super(LinearClassifierHead, self).__init__()
        
        self.head = nn.Linear(2048, cfg.MODEL.NUM_CLASSES, bias=True)
    
    def forward(self, x):
        return self.head(x)