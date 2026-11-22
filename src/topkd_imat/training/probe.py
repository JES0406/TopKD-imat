"""Probe Head: a linear classifier on frozen Student features (Top x label-free only).

Fitted with full-batch L-BFGS on standardised features of the last Layer Pair, computed
once on the training split with test-time transforms.
"""

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def collect(model: nn.Module, loader, device) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run ``model`` over ``loader``; return the per-Layer-Pair clouds and the labels (CPU)."""
    model.eval()
    feats, labels = None, []
    for x, y in loader:
        _, f = model(x.to(device, non_blocking=True))
        if feats is None:
            feats = [[] for _ in f]
        for store, cloud in zip(feats, f, strict=True):
            store.append(cloud.float().cpu())
        labels.append(y)
    return [torch.cat(s) for s in feats], torch.cat(labels)


def fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weight_decay: float = 1e-4,
    max_iter: int = 200,
) -> nn.Module:
    """Return a module mapping raw features to logits (standardisation included)."""
    mean, std = features.mean(0), features.std(0).clamp_min(1e-6)
    z = (features - mean) / std
    linear = nn.Linear(features.shape[1], num_classes)
    opt = torch.optim.LBFGS(linear.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(linear(z), labels) + weight_decay * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return _Probe(linear, mean, std)


class _Probe(nn.Module):
    def __init__(self, linear: nn.Linear, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.linear = linear
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features - self.mean) / self.std)
