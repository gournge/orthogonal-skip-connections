"""Vanilla ResNet re-using the user-supplied blocks (for CIFAR)."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Type, List

from .resnet_user_blocks import Bottleneck, Block

class ResNet(nn.Module):
    """Standard ResNet; identical to the prompt code except for typed hints &
    optional hooks for orthogonality metrics."""

    def __init__(self, ResBlock: Type[nn.Module], layers: List[int], num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(ResBlock, 64, layers[0])
        self.layer2 = self._make_layer(ResBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, planes, blocks, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

def resnet18(num_classes: int = 10):
    return ResNet(Block, [2, 2, 2, 2], num_classes)

def resnet50(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
