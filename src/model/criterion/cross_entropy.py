import torch.nn as nn


class Cross_Entropy_Loss(nn.Module):
    def __init__(self, cfg):
        super(Cross_Entropy_Loss, self).__init__()
        self.cfg = cfg

    def forward(self, out, target, **kwargs):
        loss = nn.CrossEntropyLoss()(out, target)

        return {"cross_entropy_loss": loss}
