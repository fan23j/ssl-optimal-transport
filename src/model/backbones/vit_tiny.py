# --------------------------------------------------------
# References:
# https://github.com/Kedreamix/MAE-for-CIFAR
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from einops import repeat, rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)  # 打乱index
    backward_indexes = np.argsort(forward_indexes)  # 得到原来index的位置，方便进行还原
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffle(nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape  # length, batch, dim
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indexes(
            patches, forward_indexes
        )  # 随机打乱了数据的patch，这样所有的patch都被打乱了
        patches = patches[:remain_T]  # 得到未mask的pacth [T*0.25, B, C]

        return patches, forward_indexes, backward_indexes


class ViTTiny(nn.Module):
    def __init__(self, cfg):
        super(ViTTiny, self).__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, cfg.MODEL.MAE_EMBED_DIM))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros(
                (cfg.DATASET.IMAGE_SIZE // cfg.MODEL.MAE_PATCH_SIZE) ** 2,
                1,
                cfg.MODEL.MAE_EMBED_DIM,
            )
        )

        # 对patch进行shuffle 和 mask
        self.shuffle = PatchShuffle(cfg.MODEL.MAE_MASK_RATIO)

        # 这里得到一个 (3, dim, patch, patch)
        self.patchify = torch.nn.Conv2d(
            3,
            cfg.MODEL.MAE_EMBED_DIM,
            cfg.MODEL.MAE_PATCH_SIZE,
            cfg.MODEL.MAE_PATCH_SIZE,
        )

        self.transformer = torch.nn.Sequential(
            *[
                Block(cfg.MODEL.MAE_EMBED_DIM, cfg.MODEL.MAE_ENCODER_NUM_HEADS)
                for _ in range(cfg.MODEL.MAE_ENCODER_DEPTH)
            ]
        )

        # ViT的laynorm
        self.layer_norm = torch.nn.LayerNorm(cfg.MODEL.MAE_EMBED_DIM)

        self.init_weight()

    # 初始化类别编码和向量编码
    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        return features, backward_indexes


def get_vit_tiny(cfg):
    return ViTTiny(cfg)
