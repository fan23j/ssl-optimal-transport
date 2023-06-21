# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(self, cfg):
        super(MaskedAutoencoderViT, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            cfg.DATASET.IMAGE_SIZE,
            cfg.MODEL.MAE_PATCH_SIZE,
            cfg.MODEL.MAE_IN_CHANS,
            cfg.MODEL.MAE_EMBED_DIM,
        )
        num_patches = self.patch_embed.num_patches
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.MAE_EMBED_DIM))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, cfg.MODEL.MAE_EMBED_DIM),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.MODEL.MAE_EMBED_DIM,
                    cfg.MODEL.MAE_NUM_HEADS,
                    cfg.MODEL.MAE_MLP_RATIO,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=self.norm_layer,
                )
                for i in range(cfg.MODEL.MAE_DEPTH)
            ]
        )
        self.norm = self.norm_layer(cfg.MODEL.MAE_EMBED_DIM)
        # for loss calculation
        self.mask = None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        self.mask = mask
        return latent, ids_restore


def get_vit(cfg):
    model = MaskedAutoencoderViT(cfg)
    return model
