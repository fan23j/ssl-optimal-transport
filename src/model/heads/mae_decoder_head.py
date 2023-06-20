from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from functools import partial

from timm.models.vision_transformer import Block

class MAEDecoderHead(nn.Module):
    def __init__(self, cfg):
        super(MAEDecoderHead, self).__init__()

        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.decoder_embed = nn.Linear(cfg.MODEL.MAE_EMBED_DIM, cfg.MODEL.MAE_DECODER_EMBED_DIM, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.MAE_DECODER_EMBED_DIM))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, cfg.MODEL.MAE_NUM_PATCHES + 1, cfg.MAE_DECODER_EMBED_DIM), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(cfg.MODEL.MAE_DECODER_EMBED_DIM, cfg.MAE_DECODER_NUM_HEADS, cfg.MAE_MLP_RATIO, qkv_bias=True, qk_scale=None, norm_layer=self.norm_layer)
            for i in range(cfg.MAE_DECODER_DEPTH)])

        self.decoder_norm = self.norm_layer(cfg.MAE_DECODER_EMBED_DIM)
        self.decoder_pred = nn.Linear(cfg.MAE_DECODER_EMBED_DIM, cfg.MAE_PATCH_SIZE**2 * cfg.MAE_IN_CHANS, bias=True) # decoder to patch

        self.init_weights()  

    def init_weights(self):
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)
    
    def forward(self, _x):
        x, ids_restore = _x

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x