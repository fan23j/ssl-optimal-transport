import torch.nn as nn
import torch.nn.functional as F


class BCE_With_Logits_Loss(nn.Module):
    def __init__(self, cfg):
        super(BCE_With_Logits_Loss, self).__init__()

    def forward(self, preds, targets, **kwargs):
        loss = nn.BCEWithLogitsLoss()(preds, targets)

        return {"bce_with_logits_loss": loss}
