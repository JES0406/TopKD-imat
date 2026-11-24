"""Persistence diagrams, exact persistence images and the Topology Distillation Loss.

Owner: Topology subteam (W3).
Provides the Loss interface (docs/interfaces.md #2):
``loss = topology_loss(teacher_cloud, student_cloud)``.

``topology_loss`` is an ``H0PILoss``: MSE between the 1-D H0 Persistence Images of the
Teacher's and the Student's clouds (docs/adr/0004-h0-persistence-image-loss.md).
``H0MatchedLoss`` is the variant with correspondence (docs/adr/0005-h0-matched-loss.md);
pick one by name with ``make_topology_loss``. Training runs (``run --top-loss``) offer
``h0-pi-norm``, the normalised PI calibrated on the Teacher (``paper_pi``, ADR 0006),
and ``h0-matched``; ``H0PILoss`` is kept as the library default and for the contract tests.
``RandomTopLoss`` stays as a placeholder for tests that need a cheap loss.
tests/contracts/test_loss_contract.py checks whatever is exported here.
"""

from topkd_imat.topology.base_loss import BaseTopLoss, TopologyLossError
from topkd_imat.topology.distance import topology_distance
from topkd_imat.topology.h0 import H0PILoss, h0_deaths, persistence_image_1d
from topkd_imat.topology.matched import H0MatchedLoss
from topkd_imat.topology.random_loss import RandomTopLoss

LOSSES: dict[str, type[BaseTopLoss]] = {"h0-pi": H0PILoss, "h0-matched": H0MatchedLoss}
DEFAULT_LOSS = "h0-pi"

topology_loss: BaseTopLoss = H0PILoss()


def make_topology_loss(name: str = DEFAULT_LOSS) -> BaseTopLoss:
    """A Topology Distillation Loss by name: one of ``LOSSES``."""
    if name not in LOSSES:
        raise ValueError(f"Unknown topology loss {name!r}; choose one of {sorted(LOSSES)}.")
    return LOSSES[name]()


__all__ = [
    "DEFAULT_LOSS",
    "LOSSES",
    "BaseTopLoss",
    "H0MatchedLoss",
    "H0PILoss",
    "RandomTopLoss",
    "TopologyLossError",
    "h0_deaths",
    "make_topology_loss",
    "persistence_image_1d",
    "topology_distance",
    "topology_loss",
]
