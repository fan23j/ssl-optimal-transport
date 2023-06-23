from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .backbones.resnet50 import get_resnet50
from .backbones.vit_tiny import get_vit_tiny


from .heads.projection_head import ProjectionHead
from .heads.linear_classifier_head import LinearClassifierHead
from .heads.mae_decoder_head import MAEDecoderHead
from .heads.none_head import NoneHead

_backbone_factory = {
    "resnet50": get_resnet50,
    "vit_tiny": get_vit_tiny,
}

_head_factory = {
    "projection": ProjectionHead,
    "linear": LinearClassifierHead,
    "mae_decode": MAEDecoderHead,
    "none": NoneHead,
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


def load_model(out, model, optimizer, lr_scheduler, resume=False, strict=False):
    checkpoint = torch.load(out)
    checkpoint_model = torch.load(out, map_location="cpu")["state_dict"]
    model.load_state_dict(checkpoint_model, strict=strict)
    epoch = 0
    if resume:
        print("load optimizer from {}".format(out))
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("load lr_scheduler from epoch {}".format(epoch))
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch = checkpoint["epoch"]
        print("resuming training from epoch {}".format(epoch))
    return model, optimizer, lr_scheduler, epoch


def save_model(out, epoch, model, optimizer, lr_scheduler):
    if isinstance(model, torch.nn.DataParallel):
        state = {
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
    torch.save(state, out)
