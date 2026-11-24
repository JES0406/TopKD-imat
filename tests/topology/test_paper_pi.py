"""Teacher-calibrated normalised PI (h0-pi-norm, ADR 0006): calibration and gradients."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topkd_imat.topology.paper_pi import (
    BINS,
    RANGE_MAX,
    PaperH0PILoss,
    PaperPICalibration,
    _fit_one_layer,
    fit_teacher_pi_calibrations,
    paper_pi_1d,
)


def _centres(spec: PaperPICalibration) -> torch.Tensor:
    return (torch.arange(spec.bins, dtype=torch.float64) + 0.5) * spec.range_max / spec.bins


def test_compares_shape_not_scale_and_reaches_the_student():
    teacher = torch.tensor([[0.0], [1.0], [3.0], [7.0]], dtype=torch.float64)
    spec = PaperPICalibration(0.1, 1.0, 100, 2)
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
    # The clouds are normalised, so sigma and the maximum are the same.
    for field in ("sigma", "pi_max"):
        assert getattr(specs[1], field) == pytest.approx(getattr(specs[0], field), rel=1e-5)


def test_identical_clouds_give_zero_loss():
    cloud = torch.randn(16, 5, dtype=torch.float64)
    spec = PaperPICalibration(0.1, 1.0, 100, 1)
    assert PaperH0PILoss(spec)(cloud, cloud.clone()).item() == pytest.approx(0.0, abs=1e-20)


def test_weight_is_persistence_not_tanh():
    spec = PaperPICalibration(0.05, 2.0, 100, 4, range_max=4.0)
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
    spec = PaperPICalibration(0.05, 2.0, 100, 4)
    deaths = torch.tensor([0.4, 1.0, 1.3], dtype=torch.float64)
    once = paper_pi_1d(deaths, spec)
    twice = paper_pi_1d(torch.cat([deaths, deaths]), spec)
    # Dividing by each cloud's own max would make both images peak at 1. A frozen max
    # keeps the image linear in the diagram, so doubling every point doubles the image.
    torch.testing.assert_close(twice, 2 * once)
    assert twice.max() > 1.0
    # And it uses exactly the calibrated pi_max, not 1.
    raw = paper_pi_1d(deaths, PaperPICalibration(0.05, 1.0, 100, 4))
    torch.testing.assert_close(once, raw / 2.0)


def test_teacher_calibration_matches_quantile_and_dataset_max():
    batches = [np.array([0.2, 0.6, 1.0]), np.array([0.1, 0.8, 1.2])]
    spec = _fit_one_layer(batches)
    points = np.concatenate(batches)
    distances = np.abs(points[:, None] - points[None, :]).ravel()
    assert spec.sigma == pytest.approx(np.quantile(distances[distances > 1e-5], 0.2))
    # The max is over every calibration image, so exactly one of them peaks at 1.
    peaks = [paper_pi_1d(torch.from_numpy(d), spec).max().item() for d in batches]
    assert max(peaks) == pytest.approx(1.0)
    assert min(peaks) < 1.0


def test_sigma_uses_only_the_first_200_points():
    rng = np.random.default_rng(0)
    first = rng.uniform(0.0, 1.0, 200)
    batches = [first[:120], first[120:], rng.uniform(1.5, 1.9, 300)]  # far, wide points last
    spec = _fit_one_layer(batches)
    distances = np.abs(first[:, None] - first[None, :]).ravel()
    assert spec.sigma == pytest.approx(np.quantile(distances[distances > 1e-5], 0.2))


def test_grid_is_zero_to_two_with_200_bins_whatever_the_teacher():
    # Teacher deaths only up to 1.0, as on CIFAR-10: the grid is not cut at that maximum.
    spec = _fit_one_layer([np.array([0.3, 0.5, 0.7]), np.array([0.4, 0.6, 1.0])])
    assert (spec.range_max, spec.bins) == (RANGE_MAX, BINS) == (2.0, 200)
    centres = _centres(spec)
    assert centres[0] == pytest.approx(0.005) and centres[-1] == pytest.approx(1.995)


def test_a_student_death_far_beyond_the_teachers_max_is_seen_and_pulled():
    # A scratch Student's normalised deaths reach about 2 while the Teacher's stop near 1.
    spec = _fit_one_layer([np.array([0.3, 0.5, 0.7]), np.array([0.4, 0.6, 1.0])])
    student = torch.tensor([0.5, 1.9], dtype=torch.float64, requires_grad=True)
    pi = paper_pi_1d(student, spec)
    width = spec.range_max / spec.bins
    assert abs((int(pi[100:].argmax()) + 100 + 0.5) * width - 1.9) <= width / 2
    teacher_pi = paper_pi_1d(torch.tensor([0.5, 0.6], dtype=torch.float64), spec)
    (pi - teacher_pi).square().mean().backward()
    assert student.grad[1].abs() > 0  # the outlying death gets a gradient


def test_constant_teacher_diagrams_do_not_divide_by_zero():
    spec = _fit_one_layer([np.array([1.0, 1.0]), np.array([1.0, 1.0])])
    assert spec.sigma == pytest.approx(RANGE_MAX / BINS)  # no differences: one bin width
    assert spec.pi_max > 0


def test_sigma_is_never_narrower_than_a_bin():
    # Deaths 0.001 apart would give sigma 0.001, a tenth of a bin: the Gaussians would fall
    # between bin centres. The floor keeps every death visible on the grid.
    batches = [np.array([0.500, 0.501, 0.502, 0.503])]
    spec = _fit_one_layer(batches)
    assert spec.sigma == pytest.approx(RANGE_MAX / BINS)
    off_centre = torch.tensor([0.5], dtype=torch.float64)  # 0.5 lies between two centres
    # Its nearest centre is half a bin = half a sigma away: the Gaussian there is exp(-1/8).
    peak = paper_pi_1d(off_centre, spec).max() * spec.pi_max
    assert peak == pytest.approx(0.5 * np.exp(-0.125), rel=1e-6)


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
    assert specs[1].sigma == pytest.approx(specs[0].sigma, rel=1e-5)


@pytest.mark.parametrize("shuffle", [False, True])
def test_calibration_leaves_the_global_rng_untouched(shuffle):
    # A DataLoader iterator draws its base seed from the global torch RNG even without
    # shuffling, so an unforked calibration would shift every later shuffle and augmentation.
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    fit_teacher_pi_calibrations(TinyTeacher(), _loader(shuffle), "cpu", max_batches=2)
    torch.testing.assert_close(torch.rand(3), expected)
