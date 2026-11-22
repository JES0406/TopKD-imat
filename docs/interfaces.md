# Interfaces between workstreams

These three contracts are fixed so that every subteam can work in parallel, building against a stub until the real piece lands. Internals are each owner's call; changing a contract needs agreement from the subteams on both sides of it. Terms are defined in [`CONTEXT.md`](../CONTEXT.md).

## 1. Model (Models → Experiments, Story)

Every Teacher and Student is called as:

```python
logits, features = model(x)
```

- `x`: a batch of images, shape `(batch, channels, height, width)`.
- `logits`: shape `(batch, num_classes)`.
- `features`: a list with **one Latent Point Cloud per Layer Pair**, in order from shallowest to deepest. Each element has shape `(batch, dim)`; `dim` may differ between Teacher and Student and between Layer Pairs.
- The Teacher's and the Student's `features` lists have the same length; element `i` of each forms Layer Pair `i`. The last element is the last Layer Pair (the one with Class Clusters).

## 2. Topology Distillation Loss (Topology → Experiments)

```python
loss = topology_loss(teacher_cloud, student_cloud)
```

- `teacher_cloud`: shape `(batch, dim_t)`; treated as a fixed target (no gradient needed).
- `student_cloud`: shape `(batch, dim_s)`; `dim_s` may differ from `dim_t`.
- Returns a **scalar** that gradients can flow back through to `student_cloud`.
- How the Exact PI is computed and how gradients pass through it is the Topology owner's decision (see ADR 0001).

## 3. Results (Experiments → Story)

Each training run appends **one row** to a shared results table (CSV) with these columns:

| column | meaning |
|---|---|
| `dataset` | Core Dataset or the Extension Dataset's name |
| `condition` | `Baseline`, `KD`, `Top` or `KD+Top` |
| `supervision` | `label-supervised` or `label-free` |
| `init` | `scratch` or `pretrained` |
| `seed` | integer |
| `accuracy` | test accuracy of the Student, a fraction in [0, 1]. Only `Top` × `label-free` uses a Probe Head (nothing else trains its classifier); `KD` and `KD+Top` without labels keep the head KD trained |
| `topology_distance` | distance between Teacher and Student at the last Layer Pair, on the test set; `NaN` until filled in from the saved clouds |

Each run also saves, next to its row, the Student checkpoint and the Student's test-set Latent Point Clouds for every Layer Pair, so Story can plot them without re-running training. Layout (`training/results.py` reads and writes it; `tests/contracts/results_example.csv` is a stub table to build plots against):

```
<runs dir>/
  results.csv
  teacher/clouds_test.npz                 # Teacher clouds: pair_0 ... pair_{k-1}, labels
  <dataset>_<condition>_<supervision>_<init>_s<seed>/
    config.json  metrics.csv  tb/  student.pt  clouds_test.npz
```
