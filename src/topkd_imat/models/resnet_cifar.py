"""CIFAR ResNets (He et al., 2016, Sec. 4.2) that follow the Model interface.

Adapted from chenyaofo/pytorch-cifar-models, pytorch_cifar_models/resnet.py (BSD 3-Clause,
Copyright (c) 2021 chenyaofo), itself modified from torchvision's resnet.py (BSD 3-Clause,
Copyright (c) Soumith Chintala 2016). Licence texts: THIRD_PARTY_LICENSES.md in this folder.

Module and parameter names are kept identical to the original, so its released state_dicts
load with ``strict=True``. The only behavioural change is ``forward``: it returns
``logits, features`` (docs/interfaces.md #1) with one Latent Point Cloud per stage.

A CIFAR ResNet of depth 6n+2 has a 3x3 stem conv, three stages of n BasicBlocks each
(16, 32, 64 channels at 32x32, 16x16, 8x8) and a linear classifier on the global average
pool of the last stage. The Latent Point Clouds are the global average pools of the three
stage outputs, so ``features[-1]`` is exactly the classifier's input.
"""

import torch
from torch import nn

STAGES = ("layer1", "layer2", "layer3")
STAGE_WIDTHS = (16, 32, 64)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 convs with a residual connection: ``relu(F(x) + shortcut(x))``."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CifarResNet(nn.Module):
    """CIFAR ResNet with ``blocks_per_stage`` BasicBlocks in each of its three stages."""

    def __init__(self, blocks_per_stage: int, num_classes: int = 10):
        super().__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks_per_stage)
        self.layer2 = self._make_layer(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, blocks_per_stage, stride=2)
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        layers += [BasicBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    @property
    def blocks_per_stage(self) -> tuple[int, ...]:
        return tuple(len(getattr(self, s)) for s in STAGES)

    @property
    def depth(self) -> int:
        """Weighted layers: stem conv + 2 per BasicBlock + classifier (6n+2 when uniform)."""
        return 2 + 2 * sum(self.blocks_per_stage)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        features = []
        for stage in STAGES:
            x = getattr(self, stage)(x)
            features.append(x.mean(dim=(2, 3)))  # global average pool -> (batch, channels)
        return self.fc(features[-1]), features


def cifar_resnet(depth: int, num_classes: int = 10) -> CifarResNet:
    """A randomly initialised CIFAR ResNet of ``depth`` = 6n+2 (8, 20, 32, 44, 56, ...)."""
    if depth < 8 or (depth - 2) % 6:
        raise ValueError(f"CIFAR ResNet depth must be 6n+2 with n >= 1, got {depth}.")
    return CifarResNet((depth - 2) // 6, num_classes)


def prune_depth(model: CifarResNet, blocks_per_stage: int = 1) -> CifarResNet:
    """Keep the first ``blocks_per_stage`` BasicBlocks of every stage, in place.

    The first block of each stage is the one that changes resolution and width (it holds
    the downsample shortcut), so it is always kept; later blocks only add a residual
    correction and can be dropped without breaking the network. ResNet-20 with
    ``blocks_per_stage=1`` becomes exactly a ResNet-8.
    """
    if blocks_per_stage < 1:
        raise ValueError("Each stage must keep at least its first block.")
    for stage in STAGES:
        blocks = getattr(model, stage)
        if blocks_per_stage > len(blocks):
            raise ValueError(f"{stage} has {len(blocks)} blocks; cannot keep {blocks_per_stage}.")
        setattr(model, stage, blocks[:blocks_per_stage])
    return model


def replace_head(model: CifarResNet, num_classes: int, seed: int | None = None) -> CifarResNet:
    """Swap the classifier for a freshly initialised one with ``num_classes`` outputs.

    With ``seed`` the new head's random weights are the same on every machine; the global
    random state is left untouched.
    """
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
