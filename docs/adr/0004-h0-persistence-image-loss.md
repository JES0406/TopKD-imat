# H0-only Topology Distillation Loss: MSE between 1-D Persistence Images of normalised clouds

*Status: decided by the Topology owner (W3, Marcos) on 2026-11-22; the `topology_distance` part awaits Story's (W5) agreement, and the batch composition W4's (Escobar).*

The Topology Distillation Loss uses **H0 only**. At the last Layer Pair the shape we want to transfer is the Class Clusters, which are H0 information (connected components); H1 adds cost and has no expected shape to check by eye. In a Rips filtration every H0 class is born at 0, and its death times are the edge lengths of the Euclidean minimum spanning tree (MST) of the Latent Point Cloud, so the Persistence Diagram is one-dimensional and its **Persistence Image is a 1-D grid** over persistence.

**Gradient.** The combinatorial step (which MST edge kills which component) is piecewise constant and has no useful derivative, whatever library computes it. So the MST is computed on the detached distance matrix, and the deaths are read back as `D[i, j]` from the distance matrix computed in torch, which carries the gradient to the Student. Writing the filtration itself in torch would give the same gradient (autograd only passes through the selected values) at a higher cost. The deaths are checked against gudhi/ripser in tests.

**Loss.** For each Layer Pair:

1. Divide each cloud by its own mean pairwise Euclidean distance, so the loss compares shape, not scale (the Teacher's clouds are ~4× larger than an untrained Student's, ADR 0003). Features are not normalised per vector.
2. Compute the n − 1 finite H0 deaths; the infinite bar is dropped.
3. Build the Persistence Image: Gaussians centred at each death, weighted linearly by persistence (`w(p) = p`, as defined in GI Tema 1, pp. 50–51), evaluated on a 1-D grid of **100 bins** with **σ = 2 bin widths**; total mass is not normalised.
4. The grid range is fixed: **[0, 2] for every Layer Pair**. The plan was one range per Layer Pair computed from the Teacher, but the Loss interface does not tell the loss which pair a call belongs to. The normalised Teacher deaths never exceeded 1.50 at any pair (200 batches, random 64 or balanced 80), and the margin up to 2 leaves room for the Student's outliers: a Student death beyond the grid gets no gradient.
5. Loss = MSE between the Teacher's and the Student's Persistence Images.

The total topology term is the plain mean over the three Layer Pairs, as already implemented in W4 (`w_top` × mean).

We considered using the Wasserstein-2 distance between H0 diagrams directly as the loss (sorted deaths, no parameters). We keep the Persistence Image to stay with the paper, ADR 0001 and the glossary, and use Wasserstein as the evaluation metric instead, which also gives an independent check that the loss works.

**`topology_distance`** (Results column): Wasserstein-2 between the H0 diagrams of the Teacher's and the Student's last Layer Pair, on clouds normalised as above, averaged over fixed test-set batches of the training batch size (same batches and seed for every run). A single diagram of the 10 000 test points would be dominated by the short within-cluster edges. To be confirmed with W5.

**Batch composition: class-balanced batches of 80 (8 per class)**, for every Condition. Measured on the Teacher's test-set clouds (200 batches each, seed 0), the H0 target is more stable across batches than with random batches of 64 at every Layer Pair: relative Persistence Image MSE between batches −8 / −18 / −27 % (Layer Pairs 0 / 1 / 2), Wasserstein-2 −2 / −8 / −9 %. Per epoch it costs the same or less (GPU, Teacher forward + Student step with topology: ~6.8 s vs ~9.0 s), but it makes ~20 % fewer optimiser steps. The measurement cannot separate "balanced" from "larger" (80 vs 64 points); both point the same way.

## Consequences

- No change to the glossary or the interfaces: the loss satisfies `docs/interfaces.md` #2.
- `cdist` gets an ε inside the square root, so coincident points do not produce NaN gradients.
- Resolution, σ and the grid range are configurable; the values above are the starting point, not tuned.
- H0 sees only the scales at which points merge, not the shape of each cluster or which cluster is near which. Two clouds with the same MST edge lengths have the same Persistence Image; this is a limitation to state in the memoria.
- At the first two Layer Pairs there are no Class Clusters, so the loss there matches distance distributions with no visual interpretation (as ADR 0001 anticipated).
- Cost: the topology term adds ~4.5–6 ms per batch on GPU for the three Layer Pairs (mostly the GPU→CPU copy for the MST), roughly doubling a Student step; memory is negligible.
- Deaths shrink as the batch grows. On the test set without augmentation, the maximum normalised Teacher death with balanced batches of 80 is 1.29 / 1.20 / 0.91 (Layer Pairs 0 / 1 / 2), well inside [0, 2]; it has not been checked on the training split with augmentation.
- No clear jump between the 9th and 10th longest death appears in the Teacher's last pair (median relative gap ~3 %), although cutting the 9 longest edges leaves class-pure groups (purity ~0.85). The loss sees the Class Clusters through the distribution of merge scales, not as a count of 10.
- For tuning `w_top`: the Teacher-vs-Teacher Persistence Image MSE between batches at the last pair is ~0.53, the noise floor a Student loss cannot go below.
- W4 needs a class-balanced batch sampler for the training split (classes are not exactly balanced after the validation split) and must decide whether to keep the epoch count or scale the learning-rate schedule by steps.
- scipy (for the MST) becomes a project dependency.
- The loss lives on branch `w3-topology-loss`, based on `main`; it does not depend on the models.
