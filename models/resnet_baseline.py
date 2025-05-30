import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class ResNetBaseline(nn.Module):
    """Standard ResNet-18 for CIFAR."""

    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        # adapt first conv for CIFAR
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, 1, 1, 0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.i_downsample:
            identity = self.i_downsample(identity)
        out += identity
        return self.relu(out)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.i_downsample:
            identity = self.i_downsample(identity)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    1,
                    stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, downsample, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# Factory functions
def ResNet18(num_classes, in_ch=3, **kwargs):
    return ResNet(Block, [2, 2, 2, 2], num_classes, in_ch)


def ResNet34(num_classes, in_ch=3, **kwargs):
    return ResNet(Block, [3, 4, 6, 3], num_classes, in_ch)


def ResNet50(num_classes, in_ch=3, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_ch)


def ResNet101(num_classes, in_ch=3, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_ch)


def ResNet152(num_classes, in_ch=3, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_ch)
