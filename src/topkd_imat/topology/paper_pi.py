"""Paper-inspired, exact H0 persistence images, with teacher-only calibration.

Deliberately retains this project's 1-D H0 PI and differentiable MST (rather than
replacing them with the paper's 20x20 PI and RipsNet). Implements five specific
choices from jekim5418/TopKD/ripsnet_utils.py:

1. Raw Euclidean MST death times (no per-cloud distance normalization).
2. Weight 10*tanh(persistence); birth=0 for Vietoris-Rips H0.
3. Bandwidth = 20th percentile of nonzero pairwise differences of the first
   200 teacher diagram points (per layer), as in the official code.
4. PI divided by a *shared teacher calibration maximum* (per layer), not by
   each batch's maximum.
5. PI grid bounds set to the global minimum and maximum persistence values
   observed across the Teacher calibration diagrams (per layer), never [0, 2].

Unlike the official method, the PI is still a 1-D point-sampled curve and is
computed exactly, not approximated by RipsNet. See docs/adr/0005-paper-pi.md.
"""

from dataclasses import asdict, dataclass
from itertools import islice

import numpy as np
import torch

from topkd_imat.topology.base_loss import BaseTopLoss
from topkd_imat.topology.h0 import h0_deaths


@dataclass(frozen=True)
class PaperPICalibration:
    """Frozen parameters estimated on teacher *training* features for one layer."""

    persistence_min: float
    persistence_max: float
    sigma: float
    pi_max: float
    bins: int
    n_clouds: int

    def __post_init__(self):
        if not (self.persistence_max > self.persistence_min):
            raise ValueError("PI persistence range must have positive width")
        if self.sigma <= 0 or self.pi_max <= 0 or self.bins <= 0 or self.n_clouds <= 0:
            raise ValueError("PI calibration needs positive sigma, pi_max, bins and n_clouds")

    def to_dict(self) -> dict:
        return asdict(self)


def paper_pi_1d(deaths: torch.Tensor, calibration: PaperPICalibration) -> torch.Tensor:
    """A 1-D H0 PI on fixed teacher-fitted bounds, differentiable in deaths.

    The common Gaussian normalization constant cancels against pi_max, so it is
    omitted here. Crucially, pi_max is NOT recomputed from the Student/batch.
    """
    width = (calibration.persistence_max - calibration.persistence_min) / calibration.bins
    centres = calibration.persistence_min + (
        torch.arange(calibration.bins, device=deaths.device, dtype=deaths.dtype) + 0.5
    ) * width
    gaussians = torch.exp(-0.5 * ((centres[None, :] - deaths[:, None]) / calibration.sigma).square())
    weights = 10.0 * torch.tanh(deaths)
    return (weights[:, None] * gaussians).sum(dim=0) / calibration.pi_max


class PaperH0PILoss(BaseTopLoss):
    """MSE between exact H0 PIs using fixed teacher-calibrated parameters."""

    def __init__(self, calibration: PaperPICalibration):
        self.calibration = calibration

    def image(self, cloud: torch.Tensor) -> torch.Tensor:
        return paper_pi_1d(h0_deaths(cloud, normalise=False), self.calibration)

    def _compute(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.image(teacher_cloud.to(device=student_cloud.device, dtype=student_cloud.dtype))
        return (self.image(student_cloud) - target).square().mean()


def _fit_one_layer(death_batches: list[np.ndarray], bins: int) -> PaperPICalibration:
    if not death_batches:
        raise ValueError("Need teacher training batches to calibrate PIs")
    values = np.concatenate(death_batches).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite teacher persistence values in calibration")

    # Matches the published ripsnet_utils.py: pairwise (birth, persistence)
    # distances among the first 200 points, dropping zero/near-zero distances.
    first = values[:200]
    differences = np.abs(first[:, None] - first[None, :]).ravel()
    nonzero = differences[differences > 1e-5]
    # Degenerate teacher clouds still require a well-defined differentiable PI.
    sigma = float(np.quantile(nonzero, 0.2)) if nonzero.size else 1e-5
    persistence_min = float(values.min())
    persistence_max = float(values.max())
    if persistence_max <= persistence_min:
        persistence_max = persistence_min + max(sigma, 1e-5)

    # Compute the maximum over *all calibration PIs* for this teacher layer,
    # equivalent in spirit to the dataset-global MPI in the official code.
    provisional = PaperPICalibration(
        persistence_min, persistence_max, sigma, 1.0, bins, len(death_batches)
    )
    pi_max = max(
        float(paper_pi_1d(torch.from_numpy(deaths), provisional).max())
        for deaths in death_batches
    )
    if not np.isfinite(pi_max) or pi_max <= 0:
        raise ValueError("Teacher PI maximum is zero or not finite; cannot normalize")
    return PaperPICalibration(
        persistence_min, persistence_max, sigma, pi_max, bins, len(death_batches)
    )


def fit_teacher_pi_calibrations(
    teacher: torch.nn.Module,
    train_eval_loader,
    device: str | torch.device,
    *,
    max_batches: int = 64,
    bins: int = 100,
) -> list[PaperPICalibration]:
    """Fit one reusable calibration per layer, using only teacher training data.

    The caller should provide ``loaders.train_eval`` (deterministic transforms,
    no test/validation leakage). No student output is used, no gradients are
    tracked, and the teacher is forced into eval mode.
    """
    if max_batches < 1 or bins < 1:
        raise ValueError("max_batches and bins must be positive")
    teacher.eval()
    deaths_by_layer: list[list[np.ndarray]] | None = None
    with torch.no_grad():
        for x, _ in islice(train_eval_loader, max_batches):
            if x.shape[0] < 2:
                continue
            _, features = teacher(x.to(device, non_blocking=True))
            if deaths_by_layer is None:
                deaths_by_layer = [[] for _ in features]
            if len(features) != len(deaths_by_layer):
                raise ValueError("Teacher returned inconsistent number of layer features")
            for batches, features_layer in zip(deaths_by_layer, features, strict=True):
                d = h0_deaths(features_layer, normalise=False)
                batches.append(d.detach().to(device="cpu", dtype=torch.float64).numpy())
    if not deaths_by_layer:
        raise ValueError("No teacher training clouds available for PI calibration")
    return [_fit_one_layer(batches, bins) for batches in deaths_by_layer]
