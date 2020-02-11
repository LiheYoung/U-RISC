import os
import torch
from torch import nn

from encoding.nn import SyncBatchNorm

__all__ = ["ResNet", 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
    norm_layer = SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

model_urls = {
    "resnet50": "https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b509.zip",
    "resnet101": "https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.zip",
    "resnet152": "https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip"
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
        Reference:
            - He, Kaiming, et.al "Deep Residual Learning for Image Recognition", CVPR 2016
    """
    def __init__(self, block, layers, strides=(1, 2, 2, 1)):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128)
        )
        self.bn1 = norm_layer(128)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(block, 64, layers[0], stride=strides[0], dilation=1)
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=strides[1], dilation=2)
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=strides[2], dilation=2)
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=strides[3], dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, dilation=1))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, dilation=2))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        c0 = self.maxpool(x)

        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return x, c1, c2, c3, c4


def resnet18(pretrained=False, root="data/models", **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(root, "resnet18.pth")), strict=False)
    return model


def resnet34(pretrained=False, root="data/models", **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(root, "resnet34.pth")), strict=False)
    return model


def resnet50(pretrained=False, root="data/models", **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], strides=(1, 1, 2, 2), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(root, "resnet50.pth")), strict=False)
    return model


def resnet101(pretrained=False, root="data/models", **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(root, "resnet101.pth")), strict=False)
    return model


def resnet152(pretrained=False, root="data/models", **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(root, "resnet152.pth")), strict=False)
    return model
