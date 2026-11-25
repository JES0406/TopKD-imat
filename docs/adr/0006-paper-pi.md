# ADR 0006 — H0 PI of normalised clouds, calibrated on the Teacher (`h0-pi-norm`)

The training runs offer two Topology Distillation Losses (`run --top-loss`):

| `--top-loss` | Loss |
|---|---|
| `h0-pi-norm` (default) | MSE between 1-D H0 PIs of normalised clouds on a fixed `[0, 2]` grid, with σ and a maximum calibrated on the Teacher per Layer Pair (this ADR) |
| `h0-matched` | MST-edge matching ([ADR 0005](0005-h0-matched-loss.md)) |

`h0-pi-norm` is ADR 0004's loss with σ (ADR 0004: 2 bins) and the image's scale (ADR 0004:
none) fitted to the Teacher per Layer Pair; the grid stays ADR 0004's `[0, 2]`, with 200
bins instead of 100. ADR 0004's `H0PILoss` (the old `h0-pi`) stays in the
library and the contract tests, but is no longer a run option.

The calibration follows
[TopKD's implementation](https://github.com/jekim5418/TopKD/blob/main/ripsnet_utils.py),
except for the distances (point 1), the weight (point 2) and the grid (point 5):

1. **Normalised distances, not the paper's raw ones.** Each cloud is divided by its mean
   pairwise distance before the MST, as in ADR 0004 (`h0_deaths(cloud)`), for the
   calibration and for the loss. A raw-distance variant (`h0-pi-raw`) was implemented and
   dropped; see *Why not raw distances* below.
2. **Weight: kept at `w(p)=p` (ADR 0004), not the paper's.** In H0 every birth
   is zero, so the paper's `10*tanh(p)+10*log(100*b+1)` reduces to `10*tanh(p)`.
   We do not adopt it: the factor 10 cancels against the global maximum (point 4);
   on CIFAR-10 the raw Teacher deaths at the first two Layer Pairs lie in about
   [0.16, 0.92], where `tanh(p) ~ p`; and at the last pair (deaths up to about 8)
   it would only flatten the weight of the most persistent classes. Its bound
   served the paper's RipsNet, whose last layer is a sigmoid (paper, Appendix B),
   which this project does not have.
3. **Gaussian bandwidth:** per layer, take the 0.2 quantile of the nonzero
   pairwise distances among the first 200 `(birth, persistence)` points from
   Teacher training diagrams. With H0 this is `abs(p_i-p_j)`. (The paper's text says
   "the first five quantiles"; its code uses 0.2, which we follow.)
4. **Global maximum:** compute the max of the unnormalized Teacher PIs across
   calibration batches **per layer**, then divide both Teacher and Student
   PIs by that same frozen maximum; never divide by the current batch max.
   Without RipsNet this mostly balances the three Layer Pairs.
5. **Fixed grid `[0, 2]`, as in ADR 0004, with 200 bins (`RANGE_MAX`, `BINS` in
   `paper_pi.py`), not the paper's min-max of the Teacher's diagrams.** In H0 the birth
   coordinate is always zero, so only the persistence axis needs bounds. σ is floored at
   one bin width (0.01): the grid is point-sampled, so a narrower Gaussian could fall
   between two bin centres and vanish. The floor is inactive on CIFAR-10 (σ ≈ 0.03–0.04,
   3–4 bins) and only catches degenerate Teacher clouds.

   *Why not the Teacher's min-max.* The paper cuts at the bare minimum and maximum, but it
   never builds the Student's exact PI: RipsNet maps the Student's cloud to an image and
   extrapolates however it learned to. Here the Student's exact image is evaluated on the
   grid, so a Student death more than about 3σ past the grid's edge is invisible: it gets
   no gradient, and the loss does not penalise it either, although the Teacher has no mass
   there. On CIFAR-10 the Teacher's normalised deaths lie in about [0.07, 1.04] at every
   Layer Pair, while a scratch Student's reach about 2 before training. A margin of k σ
   around the Teacher's min-max was tried: with k = 3 the grid ends near 1.1 and covers
   only 87 % of a scratch Student's pair-1 deaths (0.12–1.98; 8 batches of 64), and σ is
   a spacing between neighbouring deaths (≈ 0.36 × their standard deviation for normal
   data), not a statistical bound on where the Student goes.

   `[0, 2]` needs no data to justify: after normalisation the mean pairwise distance is 1,
   so an MST edge longer than twice the mean belongs to an isolated point, and deaths are
   never negative. It covers every Student death measured above. The price is bins the
   Teacher never fills (above about 1.05), which is why the grid has 200 bins instead of
   ADR 0004's 100: σ keeps 3–4 bins, as with the Teacher-fitted range.

The PI is still represented in 1-D with point-sampled bins; the project still uses
exact differentiable MST H0, not the authors' 2-D 20x20 GUDHI PI or their RipsNet
approximation. Thus this is **not** an exact reproduction of the paper.

## Why not raw distances

With raw distances every death is a Euclidean length in the network's feature space, so its
scale is the network's. The raw variant's grid was the Teacher's min-max (point 5), and the
Teacher's trained features are spread out; a scratch Student's random features are small and bunched together. Its deaths then fall
below the Teacher's range, only Gaussian tails reach the grid, its image is almost zero and
so is the gradient. Normalising each cloud removes the absolute scale, so both clouds' deaths
land in the same range and the loss compares shape.

Measured before training (CIFAR-10, ResNet-56 Teacher calibrated on 16 batches, one training
batch of 64; share of Student deaths within the Teacher range ± 3σ, and the norm of the
gradient on the Student's cloud):

| | pair 0 | pair 1 | pair 2 |
|---|---|---|---|
| normalised (`h0-pi-norm`), pretrained | 1.00, 0.23 | 1.00, 0.15 | 0.98, 0.18 |
| normalised (`h0-pi-norm`), scratch | 0.98, 0.24 | 0.90, 0.24 | 0.98, 3.4 |
| raw, pretrained | 0.97, 0.59 | 1.00, 0.25 | 0.98, 0.18 |
| raw, scratch | 0.98, 0.30 | **0.25, 0.02** | 1.00 (all below the Teacher's minimum), **0.001** |

The scratch Student's raw pair-2 deaths (0.05–0.46) all lie below the Teacher's range
(0.62–8.14): the gradient is about 100× smaller than in every other case. With KD or CE the
Student's scale would grow and the loss would start acting late; with Top × label-free the
topology loss is the only signal, so the Student could stay stuck. The cost of normalising
is that the Student is not asked to copy the Teacher's absolute scale, only its shape.

## Calibration and comparison

`fit_teacher_pi_calibrations` uses **only `loaders.train_eval`**, with test-time
transformations, no validation/test samples, no student outputs, and no
backpropagation. It uses the first `--pi-calibration-batches` batches (64 by
default). The original generates many teacher training PIs and uses one global
max; our calibration estimates that quantity from a bounded subset of teacher
training images, separately for each of the three layer pairs.

Iterating a DataLoader draws a seed from the global torch RNG, so calibration runs
inside `torch.random.fork_rng()`: a calibrated run gets the same shuffling and
augmentation stream as any other run with the same `--seed`.

Run names and results keys do not include the loss, and older `runs/` folders hold
ADR 0004's `h0-pi` under the same names. So **every** Top or KD+Top run, the default
loss included, needs a `--runs-dir` of its own loss; `run` and `grid` refuse the shared
`runs` folder for them (the grid before its first run). A whole grid:

```bash
uv run python -m topkd_imat.training.grid --inits pretrained scratch --seeds 0 1 \
  --epochs 60 --w-top 0.1 --top-loss h0-pi-norm --runs-dir runs_h0-pi-norm
```

Since [ADR 0007](0007-last-layer-pair-only.md) the term is trained at the last Layer Pair
only (`--top-pairs last`); the calibration is still per pair, for the logs.

Always retrain a plain KD control under identical seeds and training settings.
`--top-loss` also matters without Top: every Condition builds the loss and logs it without
training on it (`train_top`, `val_top` and per Layer Pair `*_top_<i>` in `metrics.csv`; the
test set in `test_topology.json`, in batches of the training batch size), so a KD control's
topology is measured exactly as KD+Top's. Logging changes nothing in the run: the calibration
forks the RNG and the term is computed without gradient. Before this, KD runs logged
`val_top` 0; on the first search pass (seed 0) their test term was 0.31 at the last Layer
Pair against 0.02–0.04 with KD+Top, and about 0.01 at the other two pairs for both.

**Important:** a Student death above 2 would still be invisible; none was measured, but
monitor PI coverage during training. The coefficient `w_top=0.1` was tuned on ADR 0004's PI scale and **must be re-tuned**;
comparing raw loss numbers with ADR 0004's runs would be misleading.
