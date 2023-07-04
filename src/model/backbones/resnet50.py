import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        if not isinstance(x, list):
            x = self.f(x)
        else:
            idx_crops = torch.cumsum(
                torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in x]),
                    return_counts=True,
                )[1],
                0,
            )
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.f(torch.cat(x[start_idx:end_idx]).cuda(non_blocking=True))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            x = output
        feature = torch.flatten(x, start_dim=1)
        return feature


def get_resnet50(cfg):
    model = ResNet50()

    return model
