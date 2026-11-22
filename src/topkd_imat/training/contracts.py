"""Run-time check of the Model interface (docs/interfaces.md #1).

Called once at the start of every run so that an off-contract Teacher or Student fails
immediately with a message that says what is wrong, instead of deep inside training.
"""

import torch
from torch import nn

CONTRACT = "docs/interfaces.md #1"


class ModelContractError(ValueError):
    """Raised when a model does not follow ``logits, features = model(x)``."""


def check_model(model: nn.Module, x: torch.Tensor, num_classes: int, role: str = "model"):
    """Run ``model(x)`` and check its output; return ``(logits, features)``."""
    out = model(x)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise ModelContractError(
            f"{role}: model(x) must return a pair (logits, features), got "
            f"{_describe(out)} ({CONTRACT})."
        )
    logits, features = out
    batch = x.shape[0]
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (batch, num_classes):
        raise ModelContractError(
            f"{role}: logits must have shape ({batch}, {num_classes}) for a batch of {batch}, "
            f"got {_describe(logits)} ({CONTRACT})."
        )
    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ModelContractError(
            f"{role}: features must be a non-empty list with one Latent Point Cloud per Layer "
            f"Pair, got {_describe(features)} ({CONTRACT})."
        )
    for i, cloud in enumerate(features):
        if not isinstance(cloud, torch.Tensor) or cloud.dim() != 2 or cloud.shape[0] != batch:
            raise ModelContractError(
                f"{role}: features[{i}] must have shape ({batch}, dim), got {_describe(cloud)}. "
                f"Flatten conv maps (e.g. global average pool) before returning ({CONTRACT})."
            )
    return logits, list(features)


def check_model_pair(
    teacher: nn.Module, student: nn.Module, x: torch.Tensor, num_classes: int
) -> int:
    """Check Teacher and Student together; return the number of Layer Pairs."""
    _, t_feats = check_model(teacher, x, num_classes, role="Teacher")
    _, s_feats = check_model(student, x, num_classes, role="Student")
    if len(t_feats) != len(s_feats):
        raise ModelContractError(
            f"Teacher returns {len(t_feats)} Latent Point Clouds and the Student {len(s_feats)}; "
            f"element i of each forms Layer Pair i, so the lengths must match ({CONTRACT})."
        )
    return len(t_feats)


def _describe(obj) -> str:
    if isinstance(obj, torch.Tensor):
        return f"a tensor of shape {tuple(obj.shape)}"
    if isinstance(obj, (list, tuple)):
        return f"a {type(obj).__name__} of length {len(obj)}"
    return f"a {type(obj).__name__}"
