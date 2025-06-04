"""Lightweight ResNet implementation parametrised by a Skip class."""
from __future__ import annotations

import torch.nn as nn
from .skip import get_skip, BaseSkip

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, planes: int, stride: int, skip_kind: str):
        super().__init__()
        # conv→bn→relu×2
        self.conv1 = nn.Conv2d(in_channels, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.skip: BaseSkip
        if stride != 1 or in_channels != planes:
            # projection (Conv+BN) followed by custom skip
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                get_skip(skip_kind, planes),
            )
        else:
            self.skip = get_skip(skip_kind, planes)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int, skip_kind: str = "identity"):
        super().__init__()
        assert depth in {20, 32, 44, 56, 110}, "CIFAR ResNet depths only"
        n = (depth - 2) // 6  # layers per stage
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, n, 1, skip_kind)
        self.layer2 = self._make_layer(32, n, 2, skip_kind)
        self.layer3 = self._make_layer(64, n, 2, skip_kind)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride, skip_kind):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, planes, s, skip_kind))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)