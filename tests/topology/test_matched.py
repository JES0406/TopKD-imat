"""Tests for H0MatchedLoss, the H0 loss with correspondence (ADR 0005)."""

import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from topkd_imat.topology import (
    LOSSES,
    H0MatchedLoss,
    H0PILoss,
    TopologyLossError,
    make_topology_loss,
)

K, N_PER = 10, 8


def clustered_cloud(dim=16, spread=0.1, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(K, dim, generator=g) * 3
    return centres.repeat_interleave(N_PER, 0) + spread * torch.randn(K * N_PER, dim, generator=g)


LABELS = np.repeat(np.arange(K), N_PER)


def cluster_purity(cloud: torch.Tensor) -> float:
    """Cut the K - 1 longest MST edges; fraction of points whose group's majority is their label."""
    x = cloud.detach().double().numpy()
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    tree = minimum_spanning_tree(d).tocoo()
    keep = np.argsort(tree.data)[: -(K - 1)]
    pruned = np.zeros_like(d)
    pruned[tree.row[keep], tree.col[keep]] = 1
    _, groups = connected_components(pruned, directed=False)
    hits = sum(np.bincount(LABELS[groups == g]).max() for g in np.unique(groups))
    return hits / len(LABELS)


def test_zero_for_the_same_cloud_at_any_scale():
    t = clustered_cloud()
    assert H0MatchedLoss()(t, t * 0.25).item() == pytest.approx(0.0, abs=1e-10)


def test_sees_which_image_is_which_where_the_persistence_image_does_not():
    t = clustered_cloud()
    shuffled = t[torch.randperm(len(t), generator=torch.Generator().manual_seed(0))]
    # Same diagram, so the PI loss is blind to the shuffle...
    assert H0PILoss()(t, shuffled).item() == pytest.approx(0.0, abs=1e-10)
    # ...but the images no longer sit with their neighbours.
    assert H0MatchedLoss()(t, shuffled).item() > 0.1


def test_matches_the_formula_on_a_small_example():
    # Teacher on a line 0, 1, 3; Student 0, 2, 3. Mean distances: 2 and 2.
    t = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    s = torch.tensor([[0.0], [2.0], [3.0]], dtype=torch.float64)
    d_t = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]]) / 2
    d_s = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]]) / 2
    mst_t, mst_s = [(0, 1), (1, 2)], [(1, 2), (0, 1)]
    term = lambda edges: np.mean([(d_s[i, j] - d_t[i, j]) ** 2 for i, j in edges])  # noqa: E731
    expected = 0.5 * (term(mst_t) + term(mst_s))
    assert H0MatchedLoss()(t, s).item() == pytest.approx(expected, rel=1e-6)


def test_gradient_reaches_the_student_and_teacher_dim_may_differ():
    t = clustered_cloud(dim=64)
    s = torch.randn(80, 16, requires_grad=True)
    loss = H0MatchedLoss()(t, s)
    loss.backward()
    assert loss.dim() == 0 and loss.item() > 0
    assert torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0


def test_coincident_points_give_finite_gradients():
    s = torch.zeros(6, 3, requires_grad=True)
    H0MatchedLoss()(torch.randn(6, 5), s).backward()
    assert torch.isfinite(s.grad).all()


def test_off_contract_inputs_still_fail_readably():
    with pytest.raises(TopologyLossError, match="same batch size"):
        H0MatchedLoss()(torch.randn(8, 4), torch.randn(6, 4))


def test_training_a_random_cloud_recovers_the_teacher_clusters():
    teacher = clustered_cloud(dim=64)
    student = torch.randn(80, 16, generator=torch.Generator().manual_seed(1), requires_grad=True)
    assert cluster_purity(student) < 0.5
    opt = torch.optim.Adam([student], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        H0MatchedLoss()(teacher, student).backward()
        opt.step()
    assert cluster_purity(student) > 0.95


def test_losses_by_name():
    assert set(LOSSES) == {"h0-pi", "h0-matched"}
    assert isinstance(make_topology_loss("h0-matched"), H0MatchedLoss)
    assert isinstance(make_topology_loss(), H0PILoss)
    with pytest.raises(ValueError, match="Unknown topology loss"):
        make_topology_loss("h1")
