# ADR 0007 — Distil H0 at the last Layer Pair only

*Status: accepted 2026-11-24. Partly supersedes [ADR 0001](0001-exact-pi-at-several-layer-pairs.md)
(Topology Distillation Loss at several Layer Pairs).*

The Topology Distillation Loss is now applied to the **last Layer Pair only**: `run --top-pairs
last`, the default. `--top-pairs all` keeps the old mean over the three Layer Pairs. The loss
is still H0: `h0-pi-norm` ([ADR 0006](0006-paper-pi.md)) or `h0-matched`
([ADR 0005](0005-h0-matched-loss.md)). This is also what the paper does: TopKD distils at a
single layer.

Every Layer Pair's term is still logged (`train_top_<i>`, `val_top_<i>`, and
`test_topology.json`); `top` is the term at the selected pairs, the one that is trained on.

## Evidence

All numbers come from [`results/layer_topology/`](../../results/layer_topology/), made by
`scripts/analyse_layer_topology.py` from the test-set clouds of the first search pass
([`results/search_h0-pi-norm_all-pairs/`](../../results/search_h0-pi-norm_all-pairs/):
CIFAR-10, scratch Student, seed 0, 60 epochs, `h0-pi-norm` on all three pairs). The Layer
Pairs are the average pools of the three ResNet stages (16, 32 and 64 dimensions); pair 2 is
the classifier's input.

**1. Only the last pair has a topology the Student lacks.** The `h0-pi-norm` term between
Teacher and Student test clouds (batches of 64):

| | pair 0 | pair 1 | pair 2 |
|---|---|---|---|
| KD, `w_kd` 1 | 0.009 | 0.009 | **0.306** |
| KD, `w_kd` 2 | 0.020 | 0.011 | **0.285** |
| KD+Top, `w_kd` 1, `w_top` 3 | 0.009 | 0.005 | 0.029 |
| KD+Top, `w_kd` 1, `w_top` 10 | 0.005 | 0.005 | 0.018 |

KD alone already matches the Teacher at pairs 0 and 1; all the difference is at pair 2.

**2. Pairs 0 and 1 have no structure to distil, in any homology degree.** Their clouds are not
organised by class, even in the Teacher: the mean distance to the class centroid is larger
than the mean distance between centroids (1.28 and 1.21), and the nearest class centroid is
right for 38 % and 55 % of the images. Vietoris-Rips persistence of random subsets of 64 points
(H1–H4) and 150 points (H1–H3) was compared with two nulls of the same size, a Gaussian with
the cloud's covariance and a mixture of one Gaussian per class. At pairs 0 and 1 the most
persistent H1, H2 and H3 features are 0.56–1.19 times the nulls', and exceed them in 0–80 % of
the subsets with no pattern: sampling noise. H4 is empty, in the nulls too. The real clouds
even have *fewer* H2 and H3 features than the nulls. Beyond their covariance, which a
normalised H0 diagram does not encode, there is nothing topological to copy there.

**3. Pair 2 holds the Teacher's class topology, and it is H0.** The Teacher's last cloud is 10
Class Clusters: the 9 longest MST edges per batch are 2.8 times the rest, the nearest class
centroid is right for 94 % of the images, and 92 % of the shorter MST edges join two images of
the same class. Its H1 is the only higher-degree signal found: with 150 points its most
persistent loop is 2 times the class mixture's in every subset, but short (0.16, against about
0.65 for the long H0 edges) and only about 9 loops in a batch of 64. H2 and H3 are not
consistent across sizes. No Student copies that H1.

So training pairs 0 and 1 only spends gradient on matching noise, and an H1 or H2 loss would
add little: H0 at the last pair is where the Teacher's topology is.

## What this does not fix

The same analysis shows a separate problem with `h0-pi-norm` at pair 2. KD+Top
(`w_top` 3 and 10, `w_kd` 1 and 2) copies the Teacher's H0 profile (9 long edges over the
rest: 2.2–3.0, the Teacher's 2.8) but not with the
right images: only 49–52 % of its shorter MST edges join images of the same class (KD: 60 %,
the Teacher: 92 %), and the nearest class centroid falls from 81 % (KD) to 59–64 %. A
Persistence Image does not know which point is which, so the Student forms clusters that mix
classes. This is the limitation ADR 0005 was written for; `h0-matched` is the loss that
addresses it, and it applies at the last pair the same way.

## Consequences

- `w_top` changes meaning: with `--top-pairs all` the last pair weighed `w_top / 3`, now it
  weighs `w_top`. The search uses `w_top` 0.3 / 1 / 3, the same weight on the last pair as
  1 / 3 / 10 before; its folders are named `h0-pi-norm-last_wkd<k>_wtop<w>`. The same
  search runs `h0-matched` at the last pair (`h0-matched-last_*`, `w_top` 1 / 3 / 10).
- Run names do not include `--top-pairs`: Top runs with different settings need different
  `--runs-dir`s, as for `--top-loss`.
- Training is a little cheaper: pairs 0 and 1 are computed without gradient, for the logs.
- Limits of the evidence: one seed; the clouds are average pools, so any structure inside the
  spatial feature maps is not seen; Euclidean distances only.
