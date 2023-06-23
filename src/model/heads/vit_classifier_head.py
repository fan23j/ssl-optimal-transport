# --------------------------------------------------------
# References:
# https://github.com/Kedreamix/MAE-for-CIFAR
# --------------------------------------------------------
import torch.nn as nn
import torch

from einops import rearrange

from ..backbones.vit_tiny import ViTTiny


class ViTClassifierHead(nn.Module):
    def __init__(self, cfg, encoder: ViTTiny):
        super(ViTClassifierHead, self).__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], cfg.MODEL.NUM_CLASSES)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding
        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")
        logits = self.head(features[0])
        return logits
