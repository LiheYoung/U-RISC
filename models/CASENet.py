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


class CASENet(BaseNet):
    """
        Reference:
            - Yu, Zhiding, et.al "CASENet: Deep Category-Aware Semantic Edge Detection", CVPR 2017

        CASENet is modified:
            1. All side features is 1-channel, since U-RISC is a binary classification task.
            2. Perform feature extraction on outputs of all 5 stages in ResNet, instead of 4.
            3. The backbone ResNet is modified, following 'Pytorch-Encoding'.
            4. The top stage supervision is removed, following results in DFF.
    """
    def __init__(self, backbone):
        super(CASENet, self).__init__(backbone)
        self.side0 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                                   norm_layer(1))

        self.side1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                                   norm_layer(1))

        self.side2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
                                   norm_layer(1))

        self.side3 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
                                   norm_layer(1))

        self.side4 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1),
                                   norm_layer(1))

        self.bn = norm_layer(5)
        self.classifier = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        x, c1, c2, c3, c4 = self.backbone.base_forward(x)

        side0 = self.side0(x)
        side1 = interpolate(self.side1(c1), size=(h, w), mode="bilinear", align_corners=True)
        side2 = interpolate(self.side2(c2), size=(h, w), mode="bilinear", align_corners=True)
        side3 = interpolate(self.side3(c3), size=(h, w), mode="bilinear", align_corners=True)
        side4 = interpolate(self.side4(c4), size=(h, w), mode="bilinear", align_corners=True)

        fused = torch.cat((side0, side1, side2, side3, side4), dim=1)
        fused = self.bn(fused)

        out = self.classifier(fused)
        out = self.sigmoid(out)

        return out
