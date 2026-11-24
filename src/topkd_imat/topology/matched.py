"""H0 loss with correspondence: the Student must keep the Teacher's MST edges, image by image.

Design and reasons: docs/adr/0005-h0-matched-loss.md.

``H0PILoss`` compares Persistence Images, which do not know which point is which: a Student
whose clusters hold the wrong images can match the Teacher's image exactly. Here the
Teacher's and the Student's clouds hold the same images in the same order, so an MST edge
(i, j) of one cloud names a pair of images whose distance can be read in the other cloud.
The loss asks both clouds to agree on the distances that define each other's H0:

    L = 1/2 * [ mean_{(i,j) in MST_T} (D_S[i,j] - D_T[i,j])^2
              + mean_{(i,j) in MST_S} (D_S[i,j] - D_T[i,j])^2 ]

with D the pairwise distances of each cloud divided by their own mean (as in ADR 0004).
The Teacher's edges keep the Teacher's merges (images that are neighbours in the Teacher
stay neighbours; clusters the Teacher separates stay apart, through its long edges); the
Student's edges stop it from creating short cuts the Teacher does not have. This is the
topological loss of Moor et al., "Topological Autoencoders", ICML 2020, restricted to H0.
"""

import torch

from topkd_imat.topology.base_loss import BaseTopLoss
from topkd_imat.topology.h0 import mst_edges, normalised_distances


class H0MatchedLoss(BaseTopLoss):
    """H0 with correspondence: squared differences of normalised distances on both MSTs."""

    def _compute(self, teacher_cloud: torch.Tensor, student_cloud: torch.Tensor) -> torch.Tensor:
        d_t = normalised_distances(teacher_cloud.to(student_cloud.dtype))
        d_s = normalised_distances(student_cloud)
        terms = []
        for dist in (d_t, d_s):
            i, j = mst_edges(dist)
            i = torch.as_tensor(i, dtype=torch.long, device=d_s.device)
            j = torch.as_tensor(j, dtype=torch.long, device=d_s.device)
            terms.append((d_s[i, j] - d_t[i, j]).pow(2).mean())
        return 0.5 * (terms[0] + terms[1])
