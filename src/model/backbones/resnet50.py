import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class ResNet50Pretrain(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNet50Pretrain, self).__init__()

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
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class ResNet50Linear(nn.Module):
    def __init__(self, num_class):
        super(ResNet50Linear, self).__init__()

        # encoder
        self.f = ResNet50Pretrain().f

        # linear classifier head
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


def get_resnet_50(cfg):
    return (
        ResNet50Pretrain(feature_dim=cfg.MODEL.FEATURE_DIM)
        if cfg.TASK == "simclr_pretrain"
        else ResNet50Linear(num_class=cfg.MODEL.NUM_CLASSES)
    )
