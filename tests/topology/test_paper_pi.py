"""Paper-inspired PI calibration and gradient tests (no dataset downloads)."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topkd_imat.topology.paper_pi import (
    PaperH0PILoss,
    PaperPICalibration,
    _fit_one_layer,
    fit_teacher_pi_calibrations,
    paper_pi_1d,
)


def test_uses_raw_euclidean_deaths_and_student_gradients():
    teacher = torch.tensor([[0.0], [1.0], [3.0], [7.0]], dtype=torch.float64)
    student = (2 * teacher).clone().requires_grad_(True)
    spec = PaperPICalibration(0., 15., 1., 1., 100, 2)
    loss = PaperH0PILoss(spec)(teacher, student)
    assert loss > 0  # normalized H0 clouds would have identical topology
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0


def test_tanh_weight_and_fixed_global_max():
    spec = PaperPICalibration(0., 2., 0.05, 2., 100, 4)
    deaths = torch.tensor([1.0], dtype=torch.float64)
    pi = paper_pi_1d(deaths, spec)
    centre = (torch.arange(100, dtype=torch.float64) + .5) * .02
    peak_index = int(((centre - 1).abs()).argmin())
    expected = (10 * torch.tanh(deaths) * torch.exp(-.5 * ((centre[peak_index] - 1) / .05)**2) / 2)
    torch.testing.assert_close(pi[peak_index], expected.squeeze())
    torch.testing.assert_close(pi, paper_pi_1d(deaths, spec))


def test_teacher_calibration_matches_quantile_and_dataset_max():
    batches = [np.array([.2, .6, 1.0]), np.array([.1, .8, 1.2])]
    spec = _fit_one_layer(batches, bins=20)
    points = np.concatenate(batches)
    distances = np.abs(points[:, None] - points[None, :]).ravel()
    assert spec.sigma == pytest.approx(np.quantile(distances[distances > 1e-5], .2))
    assert spec.persistence_min == .1
    assert spec.persistence_max == 1.2
    assert max(paper_pi_1d(torch.from_numpy(d), spec).max().item() for d in batches) == pytest.approx(1)


def test_pi_grid_range_uses_teacher_diagram_min_max_not_zero_two():
    # All teacher persistence values lie above 2; a hard-coded [0, 2] grid
    # would miss the peak and provide virtually no useful signal.
    batches = [np.array([2.5, 3.1, 4.2]), np.array([2.8, 3.9, 4.0])]
    spec = _fit_one_layer(batches, bins=128)
    assert spec.persistence_min == pytest.approx(2.5)
    assert spec.persistence_max == pytest.approx(4.2)

    # A Student death is rendered on the SAME fitted teacher range, with no
    # per-batch recentering, and lands in its expected grid bin.
    student_deaths = torch.tensor([3.1], dtype=torch.float64, requires_grad=True)
    pi = paper_pi_1d(student_deaths, spec)
    width = (spec.persistence_max - spec.persistence_min) / spec.bins
    peak = spec.persistence_min + (int(pi.argmax()) + .5) * width
    assert abs(peak - 3.1) <= width / 2
    pi.sum().backward()
    assert student_deaths.grad is not None
    assert torch.isfinite(student_deaths.grad).all()


def test_constant_teacher_diagrams_do_not_divide_by_zero():
    spec = _fit_one_layer([np.array([1., 1.]), np.array([1., 1.])], 16)
    assert spec.sigma > 0 and spec.pi_max > 0


class TinyTeacher(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 2, device=x.device), [x, x * 2]


def test_calibration_returns_one_set_per_layer():
    x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    loader = DataLoader(TensorDataset(x, torch.zeros(12, dtype=torch.long)), batch_size=4)
    specs = fit_teacher_pi_calibrations(TinyTeacher(), loader, "cpu", max_batches=2)
    assert len(specs) == 2
    assert all(spec.n_clouds == 2 for spec in specs)
    assert specs[1].persistence_max == pytest.approx(2 * specs[0].persistence_max, rel=1e-5)
