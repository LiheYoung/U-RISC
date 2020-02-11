import os
import torch
from torch import nn
from torch.nn.functional import interpolate
from encoding.nn import SyncBatchNorm

from .base import BaseNet


if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
    norm_layer = SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d


class DDS(BaseNet):
    """
        Reference:
            - Yun, Liu, et.al "Semantic Edge Detection with Diverse Deep Supervision", PAMI 2018

        DDS is modified:
            1. Batch normalization is added after each convolution in Residual Conv Block.
            2. Deep supervision is removed.
            3. To decrease memory and computational cost, channels of feature maps in Residual Conv Block
               are first decreased and then increased.
    """
    def __init__(self, backbone, down_scale=4):
        super(DDS, self).__init__(backbone)
        filters = [128, 256, 512, 1024, 2048]
        self.convert0 = InfoConverter(in_channels=filters[0], inter_channels=filters[0] // down_scale)
        self.convert1 = InfoConverter(in_channels=filters[1], inter_channels=filters[1] // down_scale)
        self.convert2 = InfoConverter(in_channels=filters[2], inter_channels=filters[2] // down_scale)
        self.convert3 = InfoConverter(in_channels=filters[3], inter_channels=filters[3] // down_scale)
        self.convert4 = InfoConverter(in_channels=filters[4], inter_channels=filters[4] // down_scale)

        self.bn = norm_layer(5)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        x, c1, c2, c3, c4 = self.backbone.base_forward(x)

        side0 = self.convert0(x)
        side1 = interpolate(self.convert1(c1), size=(h, w), mode="bilinear", align_corners=True)
        side2 = interpolate(self.convert2(c2), size=(h, w), mode="bilinear", align_corners=True)
        side3 = interpolate(self.convert3(c3), size=(h, w), mode="bilinear", align_corners=True)
        side4 = interpolate(self.convert4(c4), size=(h, w), mode="bilinear", align_corners=True)

        fused = torch.cat((side0, side1, side2, side3, side4), dim=1)
        fused = self.bn(fused)

        out = self.classifier(fused)
        out = self.sigmoid(out)

        return out


class InfoConverter(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(InfoConverter, self).__init__()
        self.residual_conv1 = ResidualConv(in_channels, inter_channels)
        self.residual_conv2 = ResidualConv(in_channels, inter_channels)
        self.decrease_channel = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.residual_conv1(x)
        out1 += x
        out1 = self.relu(out1)
        out = self.residual_conv2(out1)
        out += out1
        out = self.relu(out)
        out = self.decrease_channel(out)
        return out


class ResidualConv(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out

