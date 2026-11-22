"""Placeholder Topology Distillation Loss: a random value with a real gradient path.

Lets Experiments run the Top and KD+Top Conditions end to end before the real loss lands.
Any number it produces is meaningless.
"""

import warnings

import torch

from topkd_imat.topology.base_loss import BaseTopLoss


class RandomTopLoss(BaseTopLoss):
    """Returns a value drawn uniformly from [0, 1) that still backpropagates to the Student.

    The value is ``u * m / m.detach()`` with ``m`` the mean squared norm of ``student_cloud``:
    it equals ``u`` but has a non-zero gradient, so the training loop is exercised for real.
    """

    def __init__(self, seed: int | None = None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._warned = False

    def _compute(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        if not self._warned:
            warnings.warn(
                "RandomTopLoss is a placeholder: Top / KD+Top results are meaningless until the "
                "real Topology Distillation Loss replaces it in topkd_imat.topology.",
                UserWarning,
                stacklevel=3,
            )
            self._warned = True
        u = torch.rand((), generator=self._generator).to(student_cloud.device, student_cloud.dtype)
        m = student_cloud.pow(2).sum(dim=1).mean()
        return u * m / (m.detach() + 1e-12)
