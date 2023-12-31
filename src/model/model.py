from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import clip

from .backbones.resnet50 import get_resnet50
from .backbones.vit_tiny import get_vit_tiny

from .heads.projection_head import ProjectionHead
from .heads.linear_classifier_head import LinearClassifierHead
from .heads.mae_decoder_head import MAEDecoderHead
from .heads.none_head import NoneHead
from .heads.vit_classifier_head import ViTClassifierHead
from .heads.reshape_head import ReshapeHead


def get_resnet50_imagenet(cfg):
    return torch.hub.load("facebookresearch/swav:main", "resnet50")


def get_clip_b16(cfg=None):
    model, _ = clip.load("ViT-B/16", jit=False)
    return model.float()


_backbone_factory = {
    "resnet50": get_resnet50,
    "vit_tiny": get_vit_tiny,
    "resnet50_imagenet": get_resnet50_imagenet,
    "clip": get_clip_b16,
}

_head_factory = {
    "projection": ProjectionHead,
    "linear": LinearClassifierHead,
    "mae_decode": MAEDecoderHead,
    "vit_classifier": ViTClassifierHead,
    "reshape": ReshapeHead,
    "none": NoneHead,
}


class BackBone(nn.Module):
    def __init__(self, arch, cfg):
        super(BackBone, self).__init__()
        backbone = _backbone_factory[arch]
        self.backbone_model = backbone(cfg=cfg)
        self.heads = nn.ModuleList(
            [_head_factory[head_name](cfg) for head_name in cfg.MODEL.HEADS]
        )
        # self.labels_proj_head = nn.Sequential(
        #     nn.Linear(300, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, cfg.MODEL.OUTPUT_FEATURES),
        # )

    def forward(self, x):
        try:
            x = self.backbone_model(x)
        except TypeError:
            x = self.backbone_model.encode_image(x)
        for head in self.heads:
            x = head(x)
        return x


def create_model(arch, cfg):
    return BackBone(arch, cfg)


def load_model(cfg, model, optimizer=None, lr_scheduler=None, strict=False):
    checkpoint = torch.load(cfg.MODEL.PRETRAINED)
    try:
        checkpoint_model = torch.load(cfg.MODEL.PRETRAINED, map_location="cpu")[
            "state_dict"
        ]
        checkpoint_model = {
            k: v
            for k, v in checkpoint_model.items()
            if k not in cfg.MODEL.UNWANTED_KEYS
        }
        model.load_state_dict(checkpoint_model, strict=strict)
    except:
        model.load_state_dict(checkpoint, strict=strict)
    epoch = 0
    if cfg.TRAIN.RESUME:
        epoch = checkpoint["epoch"]
        print("resuming training from epoch {}".format(epoch))
        print("load optimizer from epoch {}".format(epoch))
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("load lr_scheduler from epoch {}".format(epoch))
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
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
