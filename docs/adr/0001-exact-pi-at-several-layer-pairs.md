# Exact Persistence Images at several Layer Pairs, no RipsNet

TopKD (Kim et al., 2024) computes the Topology Distillation Loss at a single layer and approximates the Persistence Images with a RipsNet, because computing them exactly on CIFAR-100/ImageNet per batch is too slow. On MNIST-scale data, we compute Exact PIs with the TDA libraries taught in the course (gudhi, ripser/persim, giotto-tda) and apply the Topology Distillation Loss at several Layer Pairs (one Student layer per block of Teacher layers, e.g. Teacher layer 5k ↔ Student layer k). Doing so removes the approximation error and the cost of training a RipsNet, and lets the Student be much shallower than the Teacher while still being guided at intermediate depths.

## Consequences

- This is a deliberate deviation from the paper: results are not a reproduction of TopKD and must be reported as an adaptation.
- How gradients reach the Student through an Exact PI is an open implementation question, owned by the task that builds the Topology Distillation Loss (not decided here).
- Only the last Layer Pair has a known expected shape (one cluster per digit); intermediate Layer Pairs can be compared numerically but not checked by eye.
