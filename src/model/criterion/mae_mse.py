import torch
import torch.nn as nn


class Mae_MSE_Loss(nn.Module):
    def __init__(self, cfg):
        super(Mae_MSE_Loss, self).__init__()
        self.mask_ratio = cfg.MODEL.MAE_MASK_RATIO

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = torch.mean((pred - imgs.to(pred.device)) ** 2 * mask) / self.mask_ratio
        return {"mae_mse_loss": loss}
