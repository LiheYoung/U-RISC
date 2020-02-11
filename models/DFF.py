import os
import torch
from torch import nn
from torch.nn.functional import interpolate, softmax
from encoding.nn import SyncBatchNorm

from .base import BaseNet


if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
    norm_layer = SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d


class DFF(BaseNet):
    """
        Reference:
            - Hu, Yuan, et.al "Dynamic Feature Fusion for Semantic Edge Detection", AAAI 2019
    """
    def __init__(self, backbone):
        super(DFF, self).__init__(backbone)
        self.side0 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                                   norm_layer(1))
        self.side0_residual = SideResidual(in_channels=128, inter_channels=128 // 8, upsample_rate=1)

        self.side1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side1_residual = SideResidual(in_channels=256, inter_channels=256 // 8, upsample_rate=2)

        self.side2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side2_residual = SideResidual(in_channels=512, inter_channels=512 // 8, upsample_rate=2)

        self.side3 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side3_residual = SideResidual(in_channels=1024, inter_channels=1024 // 8, upsample_rate=4)

        self.side4 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 16, stride=8, padding=4, bias=False))
        self.side4_residual = SideResidual(in_channels=2048, inter_channels=2048 // 8, upsample_rate=8)

        self.side4_weight = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=5, kernel_size=1),
                                          norm_layer(5),
                                          nn.ConvTranspose2d(5, 5, 16, stride=8, padding=4, bias=False))

        self.bn = norm_layer(5)
        self.sigmoid = nn.Sigmoid()

        self.ada_learner = LocationAdaptiveLearner(1, 5, 5, norm_layer=norm_layer)

    def forward(self, x):
        x, c1, c2, c3, c4 = self.backbone.base_forward(x)

        side0 = self.side0(x) + self.side0_residual(x)
        side1 = self.side1(c1) + self.side1_residual(c1)
        side2 = self.side2(c2) + self.side2_residual(c2)
        side3 = self.side3(c3) + self.side3_residual(c3)
        side4 = self.side4(c4) + self.side4_residual(c4)

        side4_weight = self.side4_weight(c4)
        ada_weights = self.ada_learner(side4_weight)

        fused = torch.cat((side0, side1, side2, side3, side4), dim=1)
        fused = self.bn(fused)
        fused = fused.view(fused.size(0), 1, -1, fused.size(2), fused.size(3))
        fused = torch.mul(fused, ada_weights)
        fused = torch.sum(fused, 2)
        out = self.sigmoid(fused)

        return out


class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3))
        return x


class SideResidual(nn.Module):
    def __init__(self, in_channels, inter_channels, upsample_rate):
        super(SideResidual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
        self.bn1 = norm_layer(inter_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(inter_channels)
        self.conv3 = nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=1)
        self.bn3 = norm_layer(1)

        self.upsample_rate = upsample_rate
        if upsample_rate == 2:
            self.upsample = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)
        elif upsample_rate == 4:
            self.upsample = nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False)
        elif upsample_rate == 8:
            self.upsample = nn.ConvTranspose2d(1, 1, 16, stride=8, padding=4, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.upsample_rate != 1:
            out = self.upsample(out)
        return out
