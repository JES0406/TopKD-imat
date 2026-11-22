"""Tests for topology_distance and the H0 Wasserstein-2 distance (ADR 0004)."""

import numpy as np
import pytest

from topkd_imat.topology import topology_distance
from topkd_imat.topology.distance import h0_wasserstein2


def test_w2_identical_is_zero_and_symmetric():
    a, b = np.array([0.1, 0.5, 1.0]), np.array([0.2, 0.4, 1.3])
    assert h0_wasserstein2(a, a) == pytest.approx(0.0)
    assert h0_wasserstein2(a, b) == pytest.approx(h0_wasserstein2(b, a))


def test_w2_matches_points_when_close():
    # Matching 1.0 <-> 1.2 costs 0.2; sending both to the diagonal would cost sqrt(0.5^2+0.6^2).
    assert h0_wasserstein2([1.0], [1.2]) == pytest.approx(0.2)


def test_w2_uses_the_diagonal_when_cheaper():
    # 0.1 <-> 1.0 would cost 0.9; both to the diagonal costs sqrt(0.05^2 + 0.5^2) ~ 0.5025.
    assert h0_wasserstein2([0.1], [1.0]) == pytest.approx(np.hypot(0.05, 0.5))


def test_w2_handles_different_sizes():
    assert h0_wasserstein2([1.0, 2.0], [1.0]) == pytest.approx(1.0)


def test_distance_is_zero_for_the_same_cloud_at_another_scale():
    cloud = np.random.default_rng(0).normal(size=(400, 64)).astype(np.float32)
    assert topology_distance(cloud, 3 * cloud) == pytest.approx(0.0, abs=1e-6)


def test_distance_is_positive_and_accepts_different_dims():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(10, 64)) * 3
    teacher = np.repeat(centres, 40, 0) + 0.1 * rng.normal(size=(400, 64))
    student = rng.normal(size=(400, 16))
    assert topology_distance(teacher, student) > 0.05


def test_distance_rejects_mismatched_clouds():
    with pytest.raises(ValueError, match="same images"):
        topology_distance(np.zeros((10, 4)), np.zeros((8, 4)))
