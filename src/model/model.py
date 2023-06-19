from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .backbones.resnet50 import ResNet50

from .heads.projection_head import ProjectionHead
from .heads.linear_classifier_head import LinearClassifierHead

_backbone_factory = {
    'resnet50': ResNet50
}

_head_factory = {
    'projection': ProjectionHead,
    'linear': LinearClassifierHead,
}

class BackBone(nn.Module):

    def __init__(self, arch, cfg):
        super(BackBone, self).__init__()
        backbone = _backbone_factory[arch]
        self.backbone_model = backbone(cfg=cfg)
        head = _head_factory[cfg.MODEL.HEAD_NAME]
        self.head = head(cfg)

    def forward(self, x):
        x = self.backbone_model(x)
        x = self.head(x)
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