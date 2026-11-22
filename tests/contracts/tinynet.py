"""TinyNet: a minimal model that follows the Model interface (docs/interfaces.md #1).

Test-only stand-in for the Teacher and the Student; not part of the package.
"""

import torch
from torch import nn


class TinyNet(nn.Module):
    """Two conv blocks, each pooled to one Latent Point Cloud; ``width`` sets the dims."""

    def __init__(self, width: int = 8, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU())
        self.head = nn.Linear(2 * width, num_classes)

    def forward(self, x: torch.Tensor):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        f1 = h1.mean(dim=(2, 3))
        f2 = h2.mean(dim=(2, 3))
        return self.head(f2), [f1, f2]
