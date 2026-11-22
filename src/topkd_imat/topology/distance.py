"""Topology Distance between a Teacher and a Student Latent Point Cloud (Results column).

Wasserstein-2 distance between the H0 Persistence Diagrams of the two clouds, each divided by
its own mean pairwise distance, averaged over fixed batches of the test set (ADR 0004). One
diagram of all 10 000 test points would be dominated by the short within-cluster edges, so the
clouds are split into batches of the training batch size; the split is a fixed permutation
(seed 0), identical for every run, and the same images go into the Teacher's and the Student's
batch. Labels are not passed through the interface, so the batches are not class-balanced.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from topkd_imat.topology.h0 import h0_deaths

BATCH_SIZE = 80
SEED = 0


def h0_wasserstein2(deaths_a, deaths_b) -> float:
    """Wasserstein-2 distance between two H0 diagrams given by their death times.

    Points are (0, d). Ground metric L-infinity, as in gudhi's default: two points cost
    |d - d'|, a point matched to the diagonal costs d / 2. Solved exactly as an assignment
    problem with each diagram augmented by the other's diagonal projections.
    """
    a = np.asarray(deaths_a, dtype=np.float64)
    b = np.asarray(deaths_b, dtype=np.float64)
    n, m = len(a), len(b)
    forbidden = 1e18
    cost = np.zeros((n + m, n + m))
    cost[:n, :m] = (a[:, None] - b[None, :]) ** 2
    cost[:n, m:] = forbidden
    cost[n:, :m] = forbidden
    cost[np.arange(n), m + np.arange(n)] = (a / 2) ** 2  # a_i -> diagonal
    cost[n + np.arange(m), np.arange(m)] = (b / 2) ** 2  # b_j -> diagonal
    # diagonal -> diagonal costs 0 (cost[n:, m:] stays 0)
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].sum()))


def topology_distance(teacher_cloud, student_cloud, batch_size: int = BATCH_SIZE) -> float:
    """Mean H0 Wasserstein-2 over fixed batches; clouds ``(n, dim_t)``, ``(n, dim_s)``."""
    t = torch.as_tensor(np.asarray(teacher_cloud), dtype=torch.float64)
    s = torch.as_tensor(np.asarray(student_cloud), dtype=torch.float64)
    if t.dim() != 2 or s.dim() != 2 or t.shape[0] != s.shape[0]:
        raise ValueError(
            f"topology_distance needs two 2-D clouds of the same images, got shapes "
            f"{tuple(t.shape)} and {tuple(s.shape)}."
        )
    n = t.shape[0]
    if n < 2:
        raise ValueError("topology_distance needs at least 2 points.")
    order = np.random.default_rng(SEED).permutation(n)
    n_batches = max(n // batch_size, 1)
    size = n // n_batches
    distances = []
    with torch.no_grad():
        for k in range(n_batches):
            idx = torch.as_tensor(order[k * size : (k + 1) * size])
            dt, ds = h0_deaths(t[idx]).numpy(), h0_deaths(s[idx]).numpy()
            distances.append(h0_wasserstein2(dt, ds))
    return float(np.mean(distances))
