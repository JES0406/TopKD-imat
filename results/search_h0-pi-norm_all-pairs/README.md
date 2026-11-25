# First `w_kd` × `w_top` search pass: `h0-pi-norm` on all three Layer Pairs

The records of the first pass of `topkd_imat.training.search` (commit `eb51559`), before
[ADR 0007](../../docs/adr/0007-last-layer-pair-only.md) restricted the topology term to the last
Layer Pair. KD+Top × label-supervised × scratch, seed 0, 60 epochs, run on the DGX. Each folder
holds the run's `results.csv` (test accuracy) and its `config.json` and `metrics.csv`; the
checkpoints, TensorBoard logs and test clouds stay out of git (they are what
`scripts/analyse_layer_topology.py` reads, see [`../layer_topology/`](../layer_topology/)).

The topology term was the **mean over the three Layer Pairs**, so the last pair had a weight of
`w_top / 3`. The KD runs predate the logging of the topology term for every Condition: their
`val_top` is 0, not a measurement.

| Folder | `w_kd` | `w_top` | Validation accuracy, last epoch | Mean of the last 5 |
|---|---|---|---|---|
| `KD_wkd1` | 1 | — | 0.8648 | 0.8668 |
| `KD_wkd2` | 2 | — | 0.8664 | 0.8686 |
| `h0-pi-norm_wkd1_wtop3` | 1 | 3 | 0.8636 | 0.8643 |
| `h0-pi-norm_wkd1_wtop10` | 1 | 10 | 0.8594 | 0.8598 |
| `h0-pi-norm_wkd1_wtop30` | 1 | 30 | 0.8294 | 0.8308 |
| `h0-pi-norm_wkd1_wtop100` | 1 | 100 | 0.5794 | 0.5764 |
| `h0-pi-norm_wkd2_wtop3` | 2 | 3 | 0.8608 | 0.8615 |
| `h0-pi-norm_wkd2_wtop10` | 2 | 10 | 0.8574 | 0.8567 |

No KD+Top setting beat KD on validation accuracy. The search was then narrowed (30 and 100
dropped) and moved to the last Layer Pair, with `w_top` 0.3 / 1 / 3: the same weight on the
last pair as 1 / 3 / 10 here.
