import torch
from torch import nn
from .resnet import *


class BaseNet(nn.Module):
    def __init__(self, backbone):
        super(BaseNet, self).__init__()
        if backbone == "resnet18":
            self.backbone = resnet18(pretrained=True)
        elif backbone == "resnet34":
            self.backbone = resnet34(pretrained=True)
        elif backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
        elif backbone == "resnet101":
            self.backbone = resnet101(pretrained=True)
        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained=True)

    def forward(self, x):
        raise NotImplementedError("Subclasses of Base must provide a forward() method")

    # test time augmentation
    def tta_eval(self, x):
        with torch.no_grad():
            out = self.forward(x)
            origin_x = x.clone()
            x = origin_x.flip(2)
            out += self.forward(x).flip(2)
            x = origin_x.flip(3)
            out += self.forward(x).flip(3)
            x = origin_x.transpose(2, 3).flip(3)
            out += self.forward(x).flip(3).transpose(2, 3)
            x = origin_x.flip(3).transpose(2, 3)
            out += self.forward(x).transpose(2, 3).flip(3)
            x = origin_x.flip(2).flip(3)
            out += self.forward(x).flip(3).flip(2)
            out /= 6.0
        return out
