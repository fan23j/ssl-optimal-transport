from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import os

from .backbones.resnet50 import get_resnet_50

_backbone_factory = {
    'resnet50': get_resnet_50,
}

class BackBone(nn.Module):

    def __init__(self, arch, cfg):
        super(BackBone, self).__init__()
        backbone = _backbone_factory[arch]
        self.backbone_model = backbone(cfg=cfg)

    def forward(self, x):
        x = self.backbone_model(x)
        return x
    

def create_model(arch, cfg):
    return BackBone(arch, cfg)

def save_model(out, epoch, model):
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)