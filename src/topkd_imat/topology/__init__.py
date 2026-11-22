"""Persistence diagrams, exact persistence images and the Topology Distillation Loss.

Owner: Topology subteam (W3).
Provides the Loss interface (docs/interfaces.md #2):
``loss = topology_loss(teacher_cloud, student_cloud)``.

``topology_loss`` is an ``H0PILoss``: MSE between the 1-D H0 Persistence Images of the
Teacher's and the Student's clouds (docs/adr/0004-h0-persistence-image-loss.md).
``RandomTopLoss`` stays as a placeholder for tests that need a cheap loss.
tests/contracts/test_loss_contract.py checks whatever is exported here.
"""

from topkd_imat.topology.base_loss import BaseTopLoss, TopologyLossError
from topkd_imat.topology.distance import topology_distance
from topkd_imat.topology.h0 import H0PILoss, h0_deaths, persistence_image_1d
from topkd_imat.topology.random_loss import RandomTopLoss

topology_loss: BaseTopLoss = H0PILoss()

__all__ = [
    "BaseTopLoss",
    "H0PILoss",
    "RandomTopLoss",
    "TopologyLossError",
    "h0_deaths",
    "persistence_image_1d",
    "topology_distance",
    "topology_loss",
]
