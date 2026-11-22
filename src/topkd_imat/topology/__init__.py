"""Persistence diagrams, exact persistence images and the Topology Distillation Loss.

Owner: Topology subteam (W3).
Provides the Loss interface (docs/interfaces.md #2):
``loss = topology_loss(teacher_cloud, student_cloud)``.

Until the real loss lands, ``topology_loss`` is a ``RandomTopLoss`` placeholder (written by
Experiments so the training loop can run). Replace it with an instance of your own
``BaseTopLoss`` subclass; tests/contracts/test_loss_contract.py checks whatever is exported here.
"""

from topkd_imat.topology.base_loss import BaseTopLoss, TopologyLossError
from topkd_imat.topology.distance import topology_distance
from topkd_imat.topology.random_loss import RandomTopLoss

topology_loss: BaseTopLoss = RandomTopLoss()

__all__ = [
    "BaseTopLoss",
    "RandomTopLoss",
    "TopologyLossError",
    "topology_distance",
    "topology_loss",
]
