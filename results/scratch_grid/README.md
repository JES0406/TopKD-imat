# Final grid: scratch Student, both Top losses, three seeds

The experiment matrix of the team's plan (2026-11-25): every Condition with the **scratch**
Student (random weights; the Teacher stays the frozen ResNet-56), CIFAR-10, 60 epochs, seeds
0, 1 and 2, run on the DGX at commit `fed4a1b`. 45 runs:

- **Main grid, 11 configurations × 3 seeds.** Baseline (label-supervised only); KD; Top and
  KD+Top with `h0-pi-norm` ([ADR 0006](../../docs/adr/0006-paper-pi.md)) and with
  `h0-matched` ([ADR 0005](../../docs/adr/0005-h0-matched-loss.md)); each label-supervised and
  label-free. Topology term at the **last Layer Pair** only
  ([ADR 0007](../../docs/adr/0007-last-layer-pair-only.md)).
- **Ablation, 4 configurations × 3 seeds** (`*-allpairs_*`): KD+Top with the topology term
  averaged over **all three Layer Pairs** (`--top-pairs all`), `w_top` 3, which puts the same
  weight on the last pair as `w_top` 1 does with `--top-pairs last`.

Weights for every Condition: `w_kd` 2 and `w_top` 1 for both losses, the best validation
accuracy of the seed-0 search ([`../search_last-pair/`](../search_last-pair/)). In label-free
runs there is no cross-entropy, so the weights only scale the step size (and set the KD/Top
ratio in KD+Top); they were not searched again. KD temperature 4, batch 64, lr 0.1, SGD,
weight decay 5e-4, as in the rest of the project.

Checkpoints and test-set Latent Point Clouds are not in git. They stay on the DGX
(`~/TopKD-imat/runs_scratch/`), packed in `~/topkd_scratch_clouds.tgz`.

## Files

| File | What |
|---|---|
| `summary.csv` | One row per configuration: mean and standard deviation over the 3 seeds of every metric below (`<metric>_mean`, `<metric>_std`), and `n_seeds`. |
| `runs.csv` | One row per run (45): the same metrics, plus `w_kd`, `w_top`, `top_pairs`, `seed`, `epochs`, `commit`. |
| `runs/<config>_s<seed>/` | Each run's `results.csv` (Results table row), `config.json`, `metrics.csv` (per epoch: losses, accuracy, topology term per Layer Pair on train and validation) and `test_topology.json`. |
| `analysis/h0_loss.csv` | `h0-pi-norm` and `h0-matched` between each run's and the Teacher's test clouds, per Layer Pair (0, 1, 2). |
| `analysis/h0_class_structure.csv` | Class structure per run and Layer Pair, and for the Teacher (`run` = `Teacher`). |
| `probe_reference.csv` | Probe Head test accuracy of every trained Student (`bn` = `trained`), of the Teacher, and of the untrained Students. |
| `reference_probe.csv` | The untrained-Student rows of `probe_reference.csv` (the control for Top × label-free). |
| `reference_class_structure.csv` | The Teacher rows of `analysis/h0_class_structure.csv`. |
| `grid.log` | The launcher's log. |

## Metrics (`runs.csv`, `summary.csv`)

All topology and class-structure metrics are on the **test set, last Layer Pair** (the
classifier's 64-dimensional input), in batches of 64 as in training, and are computed the same
way for every run, whichever loss it was trained with. So they compare across losses; the
per-epoch `train_top`/`val_top` in `metrics.csv` are each run's own loss and only compare
within a loss (Baseline and KD log `h0-pi-norm`).

| Column | Meaning |
|---|---|
| `test_acc` | Test accuracy of the Student's head; for **Top × label-free** (head never trained) the Probe Head's. |
| `val_acc_last` | Validation accuracy (5k split held out of training) at the last epoch. |
| `probe_acc` | Probe Head (linear, L-BFGS, fitted on the training split) test accuracy, for every run: the same measure of linear separability for all. |
| `w2_distance` | Wasserstein-2 between the Student's and the Teacher's H0 diagrams (`topology_distance`, ADR 0004). |
| `h0_pi_norm` | The `h0-pi-norm` term against the Teacher, with **one** Teacher calibration for all runs. |
| `h0_matched` | The `h0-matched` term against the Teacher (needs the same images in both clouds). |
| `mst_short_same_class` | Share of the shorter MST edges (all but the 9 longest) that join two images of the same class. Teacher: 0.918. |
| `intra_over_inter` | Mean distance to the class centroid / mean distance between centroids (lower: tighter, better separated classes). Teacher: 0.241. |
| `nearest_centroid_acc` | Accuracy of assigning each image to the nearest class centroid. Teacher: 0.943. |
| `longest9_over_rest` | Mean of the 9 longest MST edges over the rest: 10 Class Clusters show as 9 long edges. Teacher: 2.85. |

Labels are used here only to measure; label-free runs never saw them in training.

## Results (mean ± std over 3 seeds; accuracies in %)

| Configuration | `test_acc` | `probe_acc` | `w2_distance` | `h0_pi_norm` | `h0_matched` | `mst_short_same_class` | `nearest_centroid_acc` |
|---|---|---|---|---|---|---|---|
| Baseline, sup | 85.30 ± 0.19 | 85.59 ± 0.15 | 2.04 | 0.25 | 0.16 | 57.4 | 78.7 |
| KD, sup | 86.10 ± 0.02 | 86.51 ± 0.24 | 2.31 | 0.28 | 0.17 | 61.1 | 81.8 |
| KD, free | 86.09 ± 0.21 | 86.66 ± 0.31 | 2.26 | 0.27 | 0.17 | 61.5 | 81.3 |
| Top `h0-pi-norm`, sup | 83.19 ± 0.58 | 83.71 ± 0.56 | 0.86 | 0.02 | 0.50 | 52.0 | 59.5 |
| Top `h0-pi-norm`, free | 21.37 ± 1.23 | 21.38 ± 1.24 | 1.13 | 0.02 | 0.79 | 13.3 | 16.6 |
| Top `h0-matched`, sup | 84.95 ± 0.27 | 85.61 ± 0.12 | 1.41 | 0.16 | 0.11 | 62.1 | 78.9 |
| Top `h0-matched`, free | 79.52 ± 0.08 | 79.52 ± 0.07 | 1.20 | 0.11 | 0.10 | 62.7 | 75.5 |
| KD+Top `h0-pi-norm`, sup | 85.62 ± 0.36 | 86.19 ± 0.54 | 1.20 | 0.04 | 0.33 | 50.0 | 61.5 |
| KD+Top `h0-pi-norm`, free | 85.62 ± 0.25 | 86.20 ± 0.21 | 1.17 | 0.03 | 0.35 | 50.5 | 60.9 |
| KD+Top `h0-matched`, sup | 85.85 ± 0.41 | 86.24 ± 0.29 | 1.96 | 0.23 | 0.14 | 63.5 | 81.4 |
| KD+Top `h0-matched`, free | 85.99 ± 0.23 | 86.50 ± 0.13 | 2.05 | 0.23 | 0.14 | 63.6 | 81.2 |
| *Ablation, all pairs:* KD+Top `h0-pi-norm`, sup | 85.79 ± 0.11 | 86.30 ± 0.19 | 1.07 | 0.04 | 0.32 | 50.6 | 60.8 |
| KD+Top `h0-pi-norm`, free | 85.79 ± 0.17 | 86.43 ± 0.22 | 1.23 | 0.03 | 0.34 | 50.1 | 59.2 |
| KD+Top `h0-matched`, sup | 85.83 ± 0.06 | 86.20 ± 0.05 | 2.02 | 0.24 | 0.14 | 63.2 | 81.1 |
| KD+Top `h0-matched`, free | 86.31 ± 0.26 | 86.84 ± 0.25 | 1.98 | 0.23 | 0.14 | 63.8 | 81.0 |

References: Teacher probe 94.29 %; untrained scratch Student probe 38.1 % (BatchNorm as built;
36.7 % recomputed), the same for seeds 0–2.

Observations, without interpretation:

- KD adds about 0.8 points over Baseline, with or without labels (86.10 / 86.09 against 85.30).
- KD+Top ends between 0.5 points below KD (`h0-pi-norm`, last pair: 85.62 against 86.10) and
  0.2 above it (`h0-matched`, all pairs, label-free: 86.31 against 86.09); the seed standard
  deviations are 0.02–0.41. The term at all three pairs gives the same as at the last one
  within 0.3 points.
- `h0-pi-norm` brings its own term and W2 to the Teacher's (0.02–0.04; W2 about 1.1–1.2 against
  KD's 2.3), while the share of same-class MST edges falls to 50 % (KD 61 %) and the nearest
  class centroid to 59–62 % (KD 81 %).
- `h0-matched` lowers its own term (0.10–0.14 against KD's 0.17) and raises the share of
  same-class MST edges to 62–64 %; nearest centroid stays at 78–81 %.
- Top alone, label-free: Probe Head 79.5 % with `h0-matched`, 21.4 % with `h0-pi-norm`, against
  38.1 % for the untrained Student.
- Top alone, label-supervised: `h0-pi-norm` 83.19 %, `h0-matched` 84.95 %, Baseline 85.30 %.

## How it was made

The launcher (kept out of the repo) ran, for each seed and configuration,
`python -m topkd_imat.training.run --init scratch --epochs 60 --w-kd 2 --w-top <1|3>
--top-loss <loss> --top-pairs <last|all> --runs-dir runs_scratch/<config>_s<seed>`, one
`--runs-dir` per run because run names do not include the loss, weights or pairs. Then, on all
45 runs: `training.fill_distance` (W2), `scripts/analyse_layer_topology.py --homology` with no
values (parts 1–2, no Rips), and `training.probe_reference --trained --seeds 0 1 2` over the
saved Students; `runs.csv` and `summary.csv` join these.
