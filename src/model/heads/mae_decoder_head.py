# --------------------------------------------------------
# References:
# https://github.com/Kedreamix/MAE-for-CIFAR
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_

from ..backbones.vit_tiny import take_indexes


class MAEDecoderHead(nn.Module):
    def __init__(self, cfg):
        super(MAEDecoderHead, self).__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, cfg.MODEL.MAE_EMBED_DIM))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros(
                (cfg.DATASET.IMAGE_SIZE // cfg.MODEL.MAE_PATCH_SIZE) ** 2 + 1,
                1,
                cfg.MODEL.MAE_EMBED_DIM,
            )
        )

        self.transformer = torch.nn.Sequential(
            *[
                Block(cfg.MODEL.MAE_EMBED_DIM, cfg.MODEL.MAE_DECODER_NUM_HEADS)
                for _ in range(cfg.MODEL.MAE_DECODER_DEPTH)
            ]
        )

        self.head = torch.nn.Linear(
            cfg.MODEL.MAE_EMBED_DIM, 3 * cfg.MODEL.MAE_PATCH_SIZE**2
        )
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=cfg.MODEL.MAE_PATCH_SIZE,
            p2=cfg.MODEL.MAE_PATCH_SIZE,
            h=cfg.DATASET.IMAGE_SIZE // cfg.MODEL.MAE_PATCH_SIZE,
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        features, backward_indexes = x

        T = features.shape[0]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding  # 加上了位置编码的信息

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature 去掉全局信息，得到图像信息

        patches = self.head(features)  # 用head得到patchs
        mask = torch.zeros_like(patches)
        mask[T:] = 1  # mask其他的像素全部设为 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)  # 得到 重构之后的 img
        mask = self.patch2img(mask)

        return img, mask
