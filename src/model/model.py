from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .backbones.resnet50 import get_resnet50
from .backbones.vit import get_vit

from .heads.projection_head import ProjectionHead
from .heads.linear_classifier_head import LinearClassifierHead
from .heads.mae_decoder_head import MAEDecoderHead

_backbone_factory = {
    "resnet50": get_resnet50,
    "vit": get_vit,
}

_head_factory = {
    "projection": ProjectionHead,
    "linear": LinearClassifierHead,
    "mae_decode": MAEDecoderHead,
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


def load_model(out, model, optimizer, strict=False):
    checkpoint = torch.load(out)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch


def save_model(out, epoch, model, optimizer):
    if isinstance(model, torch.nn.DataParallel):
        state = {
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
    torch.save(state, out)
