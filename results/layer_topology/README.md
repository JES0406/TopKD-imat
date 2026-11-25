# Topology of each Layer Pair: the evidence behind ADR 0007

Made by [`scripts/analyse_layer_topology.py`](../../scripts/analyse_layer_topology.py) from the
test-set clouds of the first search pass ([`../search_h0-pi-norm_all-pairs/`](../search_h0-pi-norm_all-pairs/):
CIFAR-10, scratch ResNet-8 Student, seed 0, 60 epochs, `h0-pi-norm` trained on all three Layer
Pairs). Reading of the numbers: [ADR 0007](../../docs/adr/0007-last-layer-pair-only.md).

```bash
uv run --group analysis python scripts/analyse_layer_topology.py --root <runs> --out results/layer_topology
```

`pair` 0, 1, 2 are the average pools of the three ResNet stages (16, 32, 64 dimensions); pair 2
is the classifier's input. `run` names the search folder; `Teacher` is the ResNet-56.

| File | Rows | Columns |
|---|---|---|
| `h0_loss.csv` | run × pair | `h0_pi_norm`: the `h0-pi-norm` term between the Teacher's and the run's test clouds, mean over 156 batches of 64 |
| `h0_class_structure.csv` | run × pair | normalised MST death quantiles; `longest9_over_rest` (10 Class Clusters show as 9 long edges); `intra_over_inter` (mean distance to the class centroid / mean distance between centroids); `nearest_centroid_acc`; `mst_short_edges_same_class` and `mst_longest_edges_cross_class` (share of MST edges joining the same class / different classes), per batch of 64 |
| `higher_homology.csv` | points × pair × run × H<sub>k</sub> | Vietoris-Rips on normalised distances of random subsets (64 points: H1–H4, 20 subsets; 150 points: H1–H3, 5 subsets): mean number of features and most persistent one, the same for a Gaussian null (the cloud's covariance) and a class-mixture null (one Gaussian per class), and the share of subsets where the real maximum exceeds each null's |

A ratio of a run's maximum to a null's is meaningless where the null has no feature
(`*_max` 0): H3 and H4 mostly.
