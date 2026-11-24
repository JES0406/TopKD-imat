"""Exact H0 persistence images of normalised clouds, calibrated on the Teacher (h0-pi-norm).

Deliberately retains this project's 1-D H0 PI and differentiable MST (rather than
replacing them with the paper's 20x20 PI and RipsNet). Takes two choices from
jekim5418/TopKD/ripsnet_utils.py and keeps the rest from ADR 0004:

1. Distances: each cloud divided by its mean pairwise distance, as in ADR 0004, not the
   paper's raw distances: a scratch Student's raw deaths fall below the Teacher's range
   and get almost no gradient (ADR 0006). The calibration uses the same deaths.
2. Weight w(p) = p, as in ADR 0004, not the paper's 10*tanh(p): a constant factor cancels
   against pi_max, and for the deaths measured here tanh(p) ~ p except at the last Layer
   Pair, where it would only down-weight the most persistent classes.
3. Bandwidth = 20th percentile of nonzero pairwise differences of the first
   200 teacher diagram points (per layer), as in the official code.
4. PI divided by a *shared teacher calibration maximum* (per layer), not by
   each batch's maximum, as in the official code.
5. Grid fixed at [0, 2] (``RANGE_MAX``), as in ADR 0004, with 200 bins, not the paper's
   min-max of the Teacher's diagrams: after normalisation the mean distance is 1, so an MST
   edge beyond 2 is an isolated point, while a scratch Student's deaths reach about 2 and
   a grid cut at the Teacher's maximum (about 1) would not see them.

Unlike the official method, the PI is still a 1-D point-sampled curve and is
computed exactly, not approximated by RipsNet. See docs/adr/0006-paper-pi.md.
"""

from dataclasses import asdict, dataclass
from itertools import islice

import numpy as np
import torch

from topkd_imat.topology.base_loss import BaseTopLoss
from topkd_imat.topology.h0 import h0_deaths

RANGE_MAX = 2.0  # grid [0, RANGE_MAX] in units of the cloud's mean pairwise distance
BINS = 200  # bin width 0.01, so sigma (about 0.03-0.04 on CIFAR-10) spans 3-4 bins


@dataclass(frozen=True)
class PaperPICalibration:
    """Frozen parameters estimated on teacher *training* features for one layer.

    All values are in units of normalised distance (cloud divided by its mean pairwise
    distance), the units of ``h0_deaths``. The grid is [0, ``range_max``].
    """

    sigma: float
    pi_max: float
    bins: int
    n_clouds: int
    range_max: float = RANGE_MAX

    def __post_init__(self):
        if self.range_max <= 0:
            raise ValueError("PI range_max must be positive")
        if self.sigma <= 0 or self.pi_max <= 0 or self.bins <= 0 or self.n_clouds <= 0:
            raise ValueError("PI calibration needs positive sigma, pi_max, bins and n_clouds")

    def to_dict(self) -> dict:
        return asdict(self)


def paper_pi_1d(deaths: torch.Tensor, calibration: PaperPICalibration) -> torch.Tensor:
    """A 1-D H0 PI on the fixed grid [0, range_max], differentiable in deaths.

    Each death d adds a Gaussian weighted by w(d) = d (ADR 0004). The common Gaussian
    normalization constant cancels against pi_max, so it is omitted here. Crucially,
    pi_max is NOT recomputed from the Student/batch.
    """
    width = calibration.range_max / calibration.bins
    centres = (
        torch.arange(calibration.bins, device=deaths.device, dtype=deaths.dtype) + 0.5
    ) * width
    z = (centres[None, :] - deaths[:, None]) / calibration.sigma
    gaussians = torch.exp(-0.5 * z.square())
    return (deaths[:, None] * gaussians).sum(dim=0) / calibration.pi_max


class PaperH0PILoss(BaseTopLoss):
    """MSE between exact H0 PIs using fixed teacher-calibrated parameters."""

    def __init__(self, calibration: PaperPICalibration):
        self.calibration = calibration

    def image(self, cloud: torch.Tensor) -> torch.Tensor:
        return paper_pi_1d(h0_deaths(cloud), self.calibration)

    def _compute(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_cloud = teacher_cloud.to(device=student_cloud.device, dtype=student_cloud.dtype)
            target = self.image(teacher_cloud)
        return (self.image(student_cloud) - target).square().mean()


def _fit_one_layer(death_batches: list[np.ndarray], bins: int = BINS) -> PaperPICalibration:
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
    # The grid is point-sampled, so a Gaussian narrower than a bin can fall between two bin
    # centres and vanish: sigma is at least one bin width (inactive on CIFAR-10, where sigma
    # is 3-4 bins), which also covers degenerate Teacher clouds with no nonzero difference.
    bin_width = RANGE_MAX / bins
    sigma = max(float(np.quantile(nonzero, 0.2)) if nonzero.size else 0.0, bin_width)

    # Compute the maximum over *all calibration PIs* for this teacher layer,
    # equivalent in spirit to the dataset-global MPI in the official code.
    n = len(death_batches)
    provisional = PaperPICalibration(sigma, 1.0, bins, n)
    pi_max = max(
        float(paper_pi_1d(torch.from_numpy(deaths), provisional).max()) for deaths in death_batches
    )
    if not np.isfinite(pi_max) or pi_max <= 0:
        raise ValueError("Teacher PI maximum is zero or not finite; cannot normalize")
    return PaperPICalibration(sigma, pi_max, bins, n)


def fit_teacher_pi_calibrations(
    teacher: torch.nn.Module,
    train_eval_loader,
    device: str | torch.device,
    *,
    max_batches: int = 64,
    bins: int = BINS,
) -> list[PaperPICalibration]:
    """Fit one reusable calibration (sigma, pi_max) per layer, on teacher training data.

    The caller should provide ``loaders.train_eval`` (deterministic transforms,
    no test/validation leakage). No student output is used, no gradients are
    tracked, and the teacher is forced into eval mode.
    """
    if max_batches < 1 or bins < 1:
        raise ValueError("max_batches and bins must be positive")
    teacher.eval()
    deaths_by_layer: list[list[np.ndarray]] | None = None
    # Iterating a DataLoader draws its base seed from the global torch RNG. Fork it, so a run
    # that calibrates sees the same shuffling and augmentation as its control at that seed.
    device = torch.device(device)
    cuda = []
    if device.type == "cuda":
        cuda = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda), torch.no_grad():
        for x, _ in islice(train_eval_loader, max_batches):
            if x.shape[0] < 2:
                continue
            _, features = teacher(x.to(device, non_blocking=True))
            if deaths_by_layer is None:
                deaths_by_layer = [[] for _ in features]
            if len(features) != len(deaths_by_layer):
                raise ValueError("Teacher returned inconsistent number of layer features")
            for batches, features_layer in zip(deaths_by_layer, features, strict=True):
                d = h0_deaths(features_layer)
                batches.append(d.detach().to(device="cpu", dtype=torch.float64).numpy())
    if not deaths_by_layer:
        raise ValueError("No teacher training clouds available for PI calibration")
    return [_fit_one_layer(batches, bins) for batches in deaths_by_layer]
