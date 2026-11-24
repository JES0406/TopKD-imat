# ADR 0006 — H0 PI of normalised clouds, calibrated on the Teacher (`h0-pi-norm`)

The training runs offer two Topology Distillation Losses (`run --top-loss`):

| `--top-loss` | Loss |
|---|---|
| `h0-pi-norm` (default) | MSE between 1-D H0 PIs of normalised clouds, with the grid range, σ and a maximum calibrated on the Teacher per Layer Pair (this ADR) |
| `h0-matched` | MST-edge matching ([ADR 0005](0005-h0-matched-loss.md)) |

`h0-pi-norm` is ADR 0004's loss with the parameters ADR 0004 fixed by hand (grid `[0, 2]`,
σ = 2 bins, no normalisation of the image) fitted to the Teacher instead, per Layer Pair —
which was ADR 0004's original plan. ADR 0004's `H0PILoss` (the old `h0-pi`) stays in the
library and the contract tests, but is no longer a run option.

The calibration follows
[TopKD's implementation](https://github.com/jekim5418/TopKD/blob/main/ripsnet_utils.py),
except for the distances (point 1) and the weight (point 2):

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
5. **Data-driven range:** use the minimum and maximum persistence values across
   all the Teacher calibration diagrams **for each layer**, rather than `[0,2]`.
   The bounds are frozen for both Teacher and Student for the full training run.
   In H0 the birth coordinate is always zero, so only the persistence axis needs
   minimum/maximum bounds. If the Teacher calibration deaths are all identical,
   use a minimal positive fallback interval so its width is never zero.

The PI is still represented in 1-D with 100 point-sampled bins; the project still uses
exact differentiable MST H0, not the authors' 2-D 20x20 GUDHI PI or their RipsNet
approximation. Thus this is **not** an exact reproduction of the paper.

## Why not raw distances

With raw distances every death is a Euclidean length in the network's feature space, so its
scale is the network's. The grid is fitted to the Teacher, whose trained features are spread
out; a scratch Student's random features are small and bunched together. Its deaths then fall
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

Always retrain a plain KD control under identical seeds and training settings;
the `--top-loss` option has no effect for conditions without Top.

**Important:** the Student's normalised deaths can still exceed the Teacher's maximum
(up to 10 % of them at pair 1 for a scratch Student, above); monitor PI coverage. The
coefficient `w_top=0.1` was tuned on ADR 0004's PI scale and **must be re-tuned**;
comparing raw loss numbers with ADR 0004's runs would be misleading.
