"""Teacher-calibrated normalised PI (h0-pi-norm, ADR 0006): calibration and gradients."""

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


def _centres(spec: PaperPICalibration) -> torch.Tensor:
    width = (spec.persistence_max - spec.persistence_min) / spec.bins
    return spec.persistence_min + (torch.arange(spec.bins, dtype=torch.float64) + 0.5) * width


def test_compares_shape_not_scale_and_reaches_the_student():
    teacher = torch.tensor([[0.0], [1.0], [3.0], [7.0]], dtype=torch.float64)
    spec = PaperPICalibration(0.0, 2.0, 0.1, 1.0, 100, 2)
    # Same shape, twice the scale: the clouds are normalised, so the images are equal.
    scaled = (2 * teacher).clone().requires_grad_(True)
    assert PaperH0PILoss(spec)(teacher, scaled).item() == pytest.approx(0.0, abs=1e-12)
    # A different shape gives a positive loss with a gradient on the Student.
    other = torch.tensor([[0.0], [3.0], [4.0], [5.0]], dtype=torch.float64, requires_grad=True)
    loss = PaperH0PILoss(spec)(teacher, other)
    assert loss.item() > 1e-3
    loss.backward()
    assert other.grad is not None and torch.isfinite(other.grad).all()
    assert other.grad.abs().sum() > 0


def test_calibration_is_the_same_for_a_rescaled_teacher_layer():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 1.0], [7.0, 2.0], [8.0, 5.0], [2.0, 9.0]])
    loader = DataLoader(TensorDataset(x, torch.zeros(6, dtype=torch.long)), batch_size=3)
    specs = fit_teacher_pi_calibrations(TinyTeacher(), loader, "cpu")
    # TinyTeacher's second layer is the first one times 2: same shape, twice the scale.
    # On raw distances its range would double; normalised, every parameter is the same.
    for field in ("persistence_min", "persistence_max", "sigma", "pi_max"):
        assert getattr(specs[1], field) == pytest.approx(getattr(specs[0], field), rel=1e-5)


def test_identical_clouds_give_zero_loss():
    cloud = torch.randn(16, 5, dtype=torch.float64)
    spec = PaperPICalibration(0.0, 5.0, 0.1, 1.0, 100, 1)
    assert PaperH0PILoss(spec)(cloud, cloud.clone()).item() == pytest.approx(0.0, abs=1e-20)


def test_weight_is_persistence_not_tanh():
    spec = PaperPICalibration(0.0, 4.0, 0.05, 2.0, 100, 4)
    deaths = torch.tensor([0.5, 3.02], dtype=torch.float64)  # both on bin centres
    pi = paper_pi_1d(deaths, spec)
    gauss = torch.exp(-0.5 * ((_centres(spec)[None, :] - deaths[:, None]) / 0.05) ** 2)
    expected = (deaths[:, None] * gauss).sum(0) / 2.0  # w(p) = p, divided by pi_max
    torch.testing.assert_close(pi, expected)
    # tanh would make the two peaks almost equal (10*tanh(3) / 10*tanh(0.5) ~ 2.2);
    # w(p) = p keeps them in the ratio of the deaths.
    near = lambda d: int((_centres(spec) - d).abs().argmin())  # noqa: E731
    assert pi[near(3.02)] / pi[near(0.5)] == pytest.approx(3.02 / 0.5, rel=1e-3)


def test_pi_max_is_frozen_not_recomputed_per_cloud():
    spec = PaperPICalibration(0.0, 2.0, 0.05, 2.0, 100, 4)
    deaths = torch.tensor([0.4, 1.0, 1.3], dtype=torch.float64)
    once = paper_pi_1d(deaths, spec)
    twice = paper_pi_1d(torch.cat([deaths, deaths]), spec)
    # Dividing by each cloud's own max would make both images peak at 1. A frozen max
    # keeps the image linear in the diagram, so doubling every point doubles the image.
    torch.testing.assert_close(twice, 2 * once)
    assert twice.max() > 1.0
    # And it uses exactly the calibrated pi_max, not 1.
    raw = paper_pi_1d(deaths, PaperPICalibration(0.0, 2.0, 0.05, 1.0, 100, 4))
    torch.testing.assert_close(once, raw / 2.0)


def test_teacher_calibration_matches_quantile_and_dataset_max():
    batches = [np.array([0.2, 0.6, 1.0]), np.array([0.1, 0.8, 1.2])]
    spec = _fit_one_layer(batches, bins=20)
    points = np.concatenate(batches)
    distances = np.abs(points[:, None] - points[None, :]).ravel()
    assert spec.sigma == pytest.approx(np.quantile(distances[distances > 1e-5], 0.2))
    assert spec.persistence_min == 0.1
    assert spec.persistence_max == 1.2
    # The max is over every calibration image, so exactly one of them peaks at 1.
    peaks = [paper_pi_1d(torch.from_numpy(d), spec).max().item() for d in batches]
    assert max(peaks) == pytest.approx(1.0)
    assert min(peaks) < 1.0


def test_sigma_uses_only_the_first_200_points():
    rng = np.random.default_rng(0)
    first = rng.uniform(0.0, 1.0, 200)
    batches = [first[:120], first[120:], rng.uniform(50.0, 100.0, 300)]  # far, wide points last
    spec = _fit_one_layer(batches, bins=20)
    distances = np.abs(first[:, None] - first[None, :]).ravel()
    assert spec.sigma == pytest.approx(np.quantile(distances[distances > 1e-5], 0.2))
    assert spec.persistence_max > 50.0  # the range, unlike sigma, uses every point


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
    peak = spec.persistence_min + (int(pi.argmax()) + 0.5) * width
    assert abs(peak - 3.1) <= width / 2
    pi.sum().backward()
    assert student_deaths.grad is not None
    assert torch.isfinite(student_deaths.grad).all()


def test_constant_teacher_diagrams_do_not_divide_by_zero():
    spec = _fit_one_layer([np.array([1.0, 1.0]), np.array([1.0, 1.0])], 16)
    assert spec.sigma > 0 and spec.pi_max > 0
    assert spec.persistence_max > spec.persistence_min


class TinyTeacher(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 2, device=x.device), [x, x * 2]


def _loader(shuffle=False):
    x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    return DataLoader(
        TensorDataset(x, torch.zeros(12, dtype=torch.long)), batch_size=4, shuffle=shuffle
    )


def test_calibration_returns_one_set_per_layer():
    specs = fit_teacher_pi_calibrations(TinyTeacher(), _loader(), "cpu", max_batches=2)
    assert len(specs) == 2
    assert all(spec.n_clouds == 2 for spec in specs)
    assert specs[1].persistence_max == pytest.approx(specs[0].persistence_max, rel=1e-5)


@pytest.mark.parametrize("shuffle", [False, True])
def test_calibration_leaves_the_global_rng_untouched(shuffle):
    # A DataLoader iterator draws its base seed from the global torch RNG even without
    # shuffling, so an unforked calibration would shift every later shuffle and augmentation.
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    fit_teacher_pi_calibrations(TinyTeacher(), _loader(shuffle), "cpu", max_batches=2)
    torch.testing.assert_close(torch.rand(3), expected)
