"""H0 persistence of a Latent Point Cloud, its 1-D Persistence Image and the loss built on them.

Design and reasons: docs/adr/0004-h0-persistence-image-loss.md.

In a Rips filtration every H0 class is born at 0 and dies at the length of an edge of the
Euclidean minimum spanning tree (MST), so the finite H0 deaths are the n - 1 MST edge lengths
(the infinite bar is dropped). Which edges form the MST is combinatorial and has no useful
derivative; it is found on detached distances, and the deaths are then read from the distance
matrix computed in torch, so gradients reach the cloud through ``D[i, j]``.
"""

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

from topkd_imat.topology.base_loss import BaseTopLoss

EPS = 1e-12  # inside the square root, so coincident points do not give NaN gradients


def normalised_distances(cloud: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances of ``cloud`` (n, dim), divided by their mean (off-diagonal)."""
    sq = (cloud[:, None, :] - cloud[None, :, :]).pow(2).sum(-1)
    dist = torch.sqrt(sq + EPS)
    n = cloud.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=cloud.device)
    return dist / dist[off_diag].mean()


def mst_edges(dist: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints (i, j) of the n - 1 edges of the MST of a dense distance matrix (no gradient)."""
    d = dist.detach().to("cpu", torch.float64).numpy()
    # scipy treats zeros as missing edges; EPS keeps every off-diagonal entry > 0.
    tree = minimum_spanning_tree(d).tocoo()
    return tree.row, tree.col


def h0_deaths(cloud: torch.Tensor, normalise: bool = True) -> torch.Tensor:
    """The n - 1 finite H0 death times of ``cloud`` (n, dim), sorted, differentiable w.r.t. it."""
    dist = (
        normalised_distances(cloud)
        if normalise
        else torch.sqrt((cloud[:, None, :] - cloud[None, :, :]).pow(2).sum(-1) + EPS)
    )
    i, j = mst_edges(dist)
    i = torch.as_tensor(i, dtype=torch.long, device=cloud.device)
    j = torch.as_tensor(j, dtype=torch.long, device=cloud.device)
    return torch.sort(dist[i, j]).values


def persistence_image_1d(
    deaths: torch.Tensor, range_max: float, bins: int = 100, sigma_bins: float = 2.0
) -> torch.Tensor:
    """1-D Persistence Image of H0 deaths: persistence-weighted Gaussians on a fixed grid.

    H0 points are (0, d), so in (birth, persistence) coordinates only persistence varies and
    the image is a vector of ``bins`` values over [0, range_max]. Weight w(p) = p (GI Tema 1,
    pp. 50-51); Gaussians are not normalised and neither is the total mass.
    """
    width = range_max / bins
    centres = (torch.arange(bins, device=deaths.device, dtype=deaths.dtype) + 0.5) * width
    sigma = sigma_bins * width
    g = torch.exp(-0.5 * ((centres[None, :] - deaths[:, None]) / sigma) ** 2)
    return (deaths[:, None] * g).sum(0)


class H0PILoss(BaseTopLoss):
    """MSE between the Teacher's and the Student's 1-D H0 Persistence Images (ADR 0004).

    Each cloud is divided by its own mean pairwise distance, so the loss compares shape, not
    scale. ``range_max`` is the same for every Layer Pair: the Loss interface does not say which
    pair a call belongs to. The default 2.0 covers every normalised Teacher death measured on
    CIFAR-10 (max 1.50 over 200 batches at any pair) with room for the Student's outliers;
    a Student death beyond the grid gets no gradient.
    """

    def __init__(self, range_max: float = 2.0, bins: int = 100, sigma_bins: float = 2.0):
        if range_max <= 0 or bins < 1 or sigma_bins <= 0:
            raise ValueError("range_max, bins and sigma_bins must be positive.")
        self.range_max = range_max
        self.bins = bins
        self.sigma_bins = sigma_bins

    def image(self, cloud: torch.Tensor) -> torch.Tensor:
        return persistence_image_1d(h0_deaths(cloud), self.range_max, self.bins, self.sigma_bins)

    def _compute(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        target = self.image(teacher_cloud.to(student_cloud.dtype))
        return (self.image(student_cloud) - target).pow(2).mean()
