# Second `w_kd` × `w_top` search pass: last Layer Pair, `h0-pi-norm` and `h0-matched`

The records of the second pass of `topkd_imat.training.search` (commit `39d0864`), after
[ADR 0007](../../docs/adr/0007-last-layer-pair-only.md): the topology term trains the **last
Layer Pair only**. KD+Top × label-supervised × scratch, seed 0, 60 epochs, run on the DGX. Each
folder holds the run's `results.csv` (test accuracy), `config.json`, `metrics.csv` (with the
topology term per Layer Pair) and `test_topology.json`; checkpoints, TensorBoard logs and test
clouds stay out of git.

The KD controls are the first pass's (KD training does not depend on the topology settings):
[`../search_h0-pi-norm_all-pairs/KD_wkd1`](../search_h0-pi-norm_all-pairs/KD_wkd1) and
[`KD_wkd2`](../search_h0-pi-norm_all-pairs/KD_wkd2). Missing from this pass: `KD_wkd4`,
`h0-pi-norm-last_wkd1_wtop3` and `h0-pi-norm-last_wkd2_wtop3`.

## Validation accuracy

Selection criterion: the last epoch (`training.search`); the mean of the last 5 epochs shows
how much one epoch moves it.

| Folder | Loss | `w_kd` | `w_top` | Last epoch | Mean of the last 5 |
|---|---|---|---|---|---|
| `KD_wkd1` (first pass) | KD | 1 | — | 0.8648 | 0.8668 |
| `h0-pi-norm-last_wkd1_wtop0.3` | h0-pi-norm | 1 | 0.3 | 0.8656 | 0.8660 |
| `h0-pi-norm-last_wkd1_wtop1` | h0-pi-norm | 1 | 1 | 0.8592 | 0.8618 |
| `h0-matched-last_wkd1_wtop1` | h0-matched | 1 | 1 | 0.8660 | 0.8669 |
| `h0-matched-last_wkd1_wtop3` | h0-matched | 1 | 3 | 0.8680 | 0.8682 |
| `h0-matched-last_wkd1_wtop10` | h0-matched | 1 | 10 | 0.8558 | 0.8566 |
| `KD_wkd2` (first pass) | KD | 2 | — | 0.8664 | 0.8686 |
| `h0-pi-norm-last_wkd2_wtop0.3` | h0-pi-norm | 2 | 0.3 | 0.8616 | 0.8634 |
| `h0-pi-norm-last_wkd2_wtop1` | h0-pi-norm | 2 | 1 | 0.8672 | 0.8658 |
| `h0-matched-last_wkd2_wtop1` | h0-matched | 2 | 1 | **0.8686** | **0.8690** |
| `h0-matched-last_wkd2_wtop3` | h0-matched | 2 | 3 | 0.8634 | 0.8626 |
| `h0-matched-last_wkd2_wtop10` | h0-matched | 2 | 10 | 0.8596 | 0.8589 |
| `h0-pi-norm-last_wkd4_wtop0.3` | h0-pi-norm | 4 | 0.3 | 0.8052 | 0.8083 |
| `h0-pi-norm-last_wkd4_wtop1` | h0-pi-norm | 4 | 1 | 0.8376 | 0.8379 |
| `h0-pi-norm-last_wkd4_wtop3` | h0-pi-norm | 4 | 3 | 0.7980 | 0.7997 |
| `h0-matched-last_wkd4_wtop1` | h0-matched | 4 | 1 | 0.8326 | 0.8332 |
| `h0-matched-last_wkd4_wtop3` | h0-matched | 4 | 3 | 0.8206 | 0.8197 |
| `h0-matched-last_wkd4_wtop10` | h0-matched | 4 | 10 | 0.8386 | 0.8399 |

Best per loss: KD `w_kd` 2 (0.8664), `h0-pi-norm` `w_kd` 2 `w_top` 1 (0.8672), `h0-matched`
`w_kd` 2 `w_top` 1 (0.8686). All three are within about 0.2 points, the size of the noise
between seeds: one seed cannot rank them. `w_kd` 4 is worse with every loss (80–84 %).

## Class structure at the last Layer Pair (`analysis/`)

`analysis/h0_loss.csv` and `analysis/h0_class_structure.csv` come from
`scripts/analyse_layer_topology.py --homology` (no Rips part) on these runs, the two KD controls
and the first pass's `h0-pi-norm_wkd1_wtop3`; columns as in
[`../layer_topology/`](../layer_topology/). At pair 2, on the test set:

| | Short MST edges within a class | Intra / inter | Nearest centroid | `h0-matched` to the Teacher | 9 longest MST edges / rest |
|---|---|---|---|---|---|
| Teacher | 92 % | 0.24 | 94 % | — | 2.85 |
| KD (`w_kd` 1, 2) | 60–61 % | 0.79–0.80 | 81–82 % | 0.175 | 1.3 |
| `h0-matched` (`w_kd` 1–2, `w_top` 1–10) | 62–65 % | 0.44–0.67 | 78–82 % | 0.10–0.15 | 1.3–1.4 |
| `h0-pi-norm` (`w_kd` 1–2, `w_top` 1) | 50–51 % | 0.74–0.75 | 58–61 % | 0.37–0.38 | 2.3–2.6 |

`h0-matched` moves the Student towards the Teacher's class structure (tighter clusters, purer
MST), by a small step; `h0-pi-norm` copies the Teacher's 9 long H0 gaps with clusters that mix
classes, and ends further from the Teacher than KD by `h0-matched`'s measure. Next: the best
setting of each loss and of KD on seeds 1 and 2.
