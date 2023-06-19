from __future__ import absolute_import, division, print_function

import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(self, cfg):
        super(ProjectionHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.MODEL.FEATURE_DIM, bias=True),
        )
    
    def forward(self, x):
        return self.head(x)