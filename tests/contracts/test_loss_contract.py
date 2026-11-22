"""Contract tests for the Topology Distillation Loss interface (docs/interfaces.md #2).

Written by: Experiments subteam (consumer). Satisfied by: Topology subteam (provider).
These tests check whatever ``topkd_imat.topology.topology_loss`` currently is: the
RandomTopLoss placeholder today, the real loss once it replaces it.
"""

import warnings

import pytest
import torch

from topkd_imat.topology import BaseTopLoss, RandomTopLoss, TopologyLossError, topology_loss


@pytest.fixture(autouse=True)
def _silence_placeholder_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RandomTopLoss is a placeholder")
        yield


def test_topology_loss_is_scalar_and_differentiable_wrt_student():
    teacher = torch.randn(32, 64)
    student = torch.randn(32, 16, requires_grad=True)
    loss = topology_loss(teacher, student)
    assert isinstance(topology_loss, BaseTopLoss)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_teacher_cloud_needs_no_gradient():
    teacher = torch.randn(16, 8, requires_grad=True)
    student = torch.randn(16, 8, requires_grad=True)
    topology_loss(teacher, student).backward()
    assert teacher.grad is None


@pytest.mark.parametrize(
    ("teacher", "student", "message"),
    [
        (torch.randn(8, 4).numpy(), torch.randn(8, 4), "must be a torch.Tensor"),
        (torch.randn(8, 4), torch.randn(8, 4, 2, 2), r"must be 2-D \(batch, dim\)"),
        (torch.randn(8, 4), torch.randn(6, 4), "same batch size"),
        (torch.randn(1, 4), torch.randn(1, 4), "at least 2"),
        (torch.randn(8, 4), torch.full((8, 4), float("nan")), "NaN/inf"),
        (torch.randn(8, 4), torch.ones(8, 4, dtype=torch.long), "floating-point"),
    ],
)
def test_off_contract_inputs_fail_with_a_readable_message(teacher, student, message):
    with pytest.raises(TopologyLossError, match=message):
        topology_loss(teacher, student)


def test_subclass_returning_no_gradient_is_caught():
    class Detached(BaseTopLoss):
        def _compute(self, teacher_cloud, student_cloud):
            return student_cloud.detach().mean()

    with pytest.raises(TopologyLossError, match="no gradient path"):
        Detached()(torch.randn(8, 4), torch.randn(8, 4, requires_grad=True))


def test_subclass_returning_a_vector_is_caught():
    class Vector(BaseTopLoss):
        def _compute(self, teacher_cloud, student_cloud):
            return student_cloud.mean(dim=1)

    with pytest.raises(TopologyLossError, match="0-dim tensor"):
        Vector()(torch.randn(8, 4), torch.randn(8, 4, requires_grad=True))


def test_random_placeholder_warns_once_and_is_reproducible_with_a_seed():
    a, b = RandomTopLoss(seed=0), RandomTopLoss(seed=0)
    x, y = torch.randn(8, 4), torch.randn(8, 3, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert a(x, y).item() == b(x, y).item()
        a(x, y)
    assert sum("placeholder" in str(w.message) for w in caught) == 2  # once per instance
