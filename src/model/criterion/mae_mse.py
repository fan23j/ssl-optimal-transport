import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed


class Mae_MSE_Loss(nn.Module):
    def __init__(self, cfg):
        super(Mae_MSE_Loss, self).__init__()
        self.norm_pix_loss = cfg.LOSS.NORM_PIX_LOSS
        self.patch_embed = PatchEmbed(
            cfg.DATASET.IMAGE_SIZE,
            cfg.MODEL.MAE_PATCH_SIZE,
            cfg.MODEL.MAE_IN_CHANS,
            cfg.MODEL.MAE_EMBED_DIM,
        )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs).to(pred.device)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + self.cfg.NORM_PIX_LOSS_EPSILON) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return {"mae_mse_loss": loss}
