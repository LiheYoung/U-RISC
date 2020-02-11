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


class ResNetUNet(BaseNet):
    """
        Reference:
            - Olaf, Ronneberger, et.al "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
            - He, Kaiming, et.al "Deep Residual Learning for Image Recognition", CVPR 2016

        U-Net is modified:
            1. Backbone network is ResNet, and output_stride = 8
    """
    def __init__(self, backbone):
        super(ResNetUNet, self).__init__(backbone)
        filters = [128, 256, 512, 1024, 2048]
        self.up1 = Up(in_channels=filters[4], out_channels=filters[3])
        self.up2 = Up(in_channels=filters[3], out_channels=filters[2])
        self.up3 = Up(in_channels=filters[2], out_channels=filters[1])
        self.up4 = Up(in_channels=filters[1], out_channels=filters[0])

        self.classfier = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c0, c1, c2, c3, c4 = self.backbone.base_forward(x)
        x = self.up1(c4, c3)
        x = self.up2(x, c2)
        x = self.up3(x, c1)
        x = self.up4(x, c0)
        x = self.classfier(x)
        x = self.sigmoid(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.decrease_channel = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.decrease_channel(x1)
        h, w = x2.shape[2], x2.shape[3]
        x1 = interpolate(x1, size=(h, w), mode="bilinear", align_corners=True)
        fused = torch.cat((x1, x2), dim=1)
        out = self.double_conv(fused)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
