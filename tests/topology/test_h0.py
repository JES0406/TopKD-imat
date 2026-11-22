"""Tests for the H0 deaths, the 1-D Persistence Image and H0PILoss (ADR 0004)."""

import numpy as np
import pytest
import torch

from topkd_imat.topology import H0PILoss, h0_deaths, persistence_image_1d, topology_loss


def kruskal_deaths(points: np.ndarray) -> np.ndarray:
    """Independent reference: MST edge lengths by Kruskal with union-find."""
    n = len(points)
    d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    edges = sorted((d[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deaths = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            deaths.append(w)
    return np.array(deaths)


def clustered_cloud(n_per=8, k=10, dim=16, spread=0.1, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(k, dim, generator=g) * 3
    pts = centres.repeat_interleave(n_per, 0) + spread * torch.randn(k * n_per, dim, generator=g)
    return pts


def test_deaths_on_a_line_are_the_gaps():
    cloud = torch.tensor([[0.0], [1.0], [3.0], [7.0]], dtype=torch.float64)
    torch.testing.assert_close(
        h0_deaths(cloud, normalise=False), torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deaths_match_kruskal(seed):
    pts = np.random.default_rng(seed).normal(size=(40, 5))
    ours = h0_deaths(torch.as_tensor(pts), normalise=False).numpy()
    np.testing.assert_allclose(ours, kruskal_deaths(pts), rtol=1e-6)


def test_deaths_match_gudhi_rips_h0():
    gudhi = pytest.importorskip("gudhi")
    pts = np.random.default_rng(3).normal(size=(30, 4))
    st = gudhi.RipsComplex(points=pts).create_simplex_tree(max_dimension=1)
    st.persistence()
    ref = np.sort(st.persistence_intervals_in_dimension(0)[:, 1])
    ref = ref[np.isfinite(ref)]
    np.testing.assert_allclose(h0_deaths(torch.as_tensor(pts), normalise=False).numpy(), ref)


def test_deaths_count_and_scale_invariance():
    cloud = torch.randn(80, 32)
    d = h0_deaths(cloud)
    assert d.shape == (79,)
    torch.testing.assert_close(h0_deaths(cloud * 7.5), d)


def test_clusters_show_as_nine_long_deaths():
    d = h0_deaths(clustered_cloud())
    assert d[-9:].min() > 5 * d[:-9].max()


def test_coincident_points_give_finite_gradients():
    cloud = torch.randn(10, 4)
    cloud[3] = cloud[7]
    cloud.requires_grad_(True)
    h0_deaths(cloud).sum().backward()
    assert torch.isfinite(cloud.grad).all()


def test_persistence_image_shape_and_peak():
    pi = persistence_image_1d(torch.tensor([1.0]), range_max=2.0, bins=100)
    assert pi.shape == (100,)
    assert pi.argmax().item() in (49, 50)
    # 1.0 lies half a bin (a quarter sigma) from the nearest centres: peak = exp(-1/32).
    torch.testing.assert_close(pi.max(), torch.tensor(np.exp(-1 / 32), dtype=torch.float32))


def test_persistence_image_is_weighted_by_persistence():
    pi = persistence_image_1d(torch.tensor([0.5, 1.5]), range_max=2.0, bins=100)
    assert pi[74].item() == pytest.approx(3 * pi[24].item(), rel=0.01)


def test_exported_loss_is_h0_pi():
    assert isinstance(topology_loss, H0PILoss)


def test_loss_is_zero_for_the_same_shape_at_any_scale():
    t = clustered_cloud()
    assert topology_loss(t, t * 0.25).item() == pytest.approx(0.0, abs=1e-10)


def test_loss_ignores_teacher_dimension():
    t = clustered_cloud(dim=64)
    s = torch.randn(80, 16, requires_grad=True)
    loss = topology_loss(t, s)
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0


def test_training_a_random_cloud_pulls_it_towards_the_teacher_shape():
    teacher = clustered_cloud(dim=64)
    student = torch.randn(80, 16, generator=torch.Generator().manual_seed(1), requires_grad=True)
    opt = torch.optim.Adam([student], lr=0.05)
    first = topology_loss(teacher, student).item()
    for _ in range(300):
        opt.zero_grad()
        loss = topology_loss(teacher, student)
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        H0PILoss(range_max=0)
