from __future__ import annotations

import torch.nn as nn
from .skip import get_skip, BaseSkip

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, skip_kind: str = "identity", downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.skip: BaseSkip = get_skip(skip_kind, planes * self.expansion)
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        identity = self.skip(identity)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1, skip_kind: str = "identity", downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.skip: BaseSkip = get_skip(skip_kind, planes * self.expansion)
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        identity = self.skip(identity)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int, skip_kind: str = "identity"):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, skip_kind=skip_kind)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, skip_kind=skip_kind)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, skip_kind=skip_kind)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, skip_kind=skip_kind)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride, skip_kind):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, skip_kind, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, skip_kind=skip_kind))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

def resnet18(num_classes: int = 1000, skip_kind: str = "identity") -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, skip_kind)

def resnet50(num_classes: int = 1000, skip_kind: str = "identity") -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, skip_kind)
