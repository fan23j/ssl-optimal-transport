import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE_Loss(nn.Module):
    def __init__(self, cfg):
        super(BCE_Loss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, preds, targets, **kwargs):
        preds = torch.sigmoid(preds)
        loss = self.loss_fn(preds, targets.float())

        return {"bce_loss": loss}
