"""Base class for Topology Distillation Losses (Loss interface, docs/interfaces.md #2).

Subclasses implement ``_compute``; ``__call__`` checks the inputs and the output so that
a shape or dtype mistake fails with a message that says what is wrong and how to fix it.
"""

from abc import ABC, abstractmethod

import torch

CONTRACT = "docs/interfaces.md #2"


class TopologyLossError(ValueError):
    """Raised when a Topology Distillation Loss gets or returns something off-contract."""


class BaseTopLoss(ABC):
    """``loss = topology_loss(teacher_cloud, student_cloud)``.

    - ``teacher_cloud``: ``(batch, dim_t)`` tensor, a fixed target (detached before ``_compute``).
    - ``student_cloud``: ``(batch, dim_s)`` tensor; ``dim_s`` may differ from ``dim_t``.
    - Returns a 0-dim tensor that gradients flow back through to ``student_cloud``.
    """

    def __call__(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        name = type(self).__name__
        _check_cloud(name, "teacher_cloud", teacher_cloud)
        _check_cloud(name, "student_cloud", student_cloud)
        if teacher_cloud.shape[0] != student_cloud.shape[0]:
            raise TopologyLossError(
                f"{name}: teacher_cloud and student_cloud must hold the same images, so the same "
                f"batch size; got {teacher_cloud.shape[0]} and {student_cloud.shape[0]}. "
                f"Pass the Teacher and the Student the same batch ({CONTRACT})."
            )
        loss = self._compute(teacher_cloud.detach(), student_cloud)
        _check_output(name, loss, student_cloud)
        return loss

    @abstractmethod
    def _compute(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        """Return a 0-dim tensor, differentiable with respect to ``student_cloud``."""


def _check_cloud(name: str, arg: str, cloud) -> None:
    if not isinstance(cloud, torch.Tensor):
        raise TopologyLossError(
            f"{name}: {arg} must be a torch.Tensor, got {type(cloud).__name__}. "
            f"Pass the Latent Point Cloud straight from the model's features list ({CONTRACT})."
        )
    if cloud.dim() != 2:
        raise TopologyLossError(
            f"{name}: {arg} must be 2-D (batch, dim), got shape {tuple(cloud.shape)}. "
            f"Flatten each Layer Pair's features to (batch, dim) inside the model "
            f"(docs/interfaces.md #1)."
        )
    if not cloud.is_floating_point():
        raise TopologyLossError(
            f"{name}: {arg} must be a floating-point tensor, got dtype {cloud.dtype}."
        )
    if cloud.shape[0] < 2:
        raise TopologyLossError(
            f"{name}: {arg} has {cloud.shape[0]} point(s); a point cloud needs at least 2 to have "
            f"any topology. Check the batch size and drop_last in the DataLoader."
        )
    if not torch.isfinite(cloud).all():
        bad = int((~torch.isfinite(cloud)).sum())
        raise TopologyLossError(
            f"{name}: {arg} contains {bad} NaN/inf value(s). The model diverged upstream of the "
            f"loss; lower the learning rate or check the inputs."
        )


def _check_output(name: str, loss, student_cloud: torch.Tensor) -> None:
    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        shape = tuple(loss.shape) if isinstance(loss, torch.Tensor) else type(loss).__name__
        raise TopologyLossError(
            f"{name}._compute must return a 0-dim tensor (a scalar), got {shape}. "
            f"Reduce with .mean() or .sum() ({CONTRACT})."
        )
    if not torch.isfinite(loss):
        raise TopologyLossError(f"{name}._compute returned {loss.item()}, not a finite number.")
    if student_cloud.requires_grad and not loss.requires_grad:
        raise TopologyLossError(
            f"{name}._compute returned a tensor with no gradient path to student_cloud, so it "
            f"cannot train the Student. Avoid .item(), .detach(), numpy or no_grad on the "
            f"student side ({CONTRACT})."
        )
