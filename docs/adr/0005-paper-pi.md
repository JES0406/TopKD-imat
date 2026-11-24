# ADR 0005 — Paper-inspired PI calibration (experimental variant)

The existing `H0PILoss` remains unchanged. Select `--pi-mode paper-h0` to test five
construction choices from [TopKD's implementation](https://github.com/jekim5418/TopKD/blob/main/ripsnet_utils.py):

1. **Raw distances:** use `h0_deaths(..., normalise=False)`, so the Euclidean MST
   edge lengths retain the Teacher/Student distance scale. No normalization by
   the mean distance of each cloud.
2. **Weight:** `10*tanh(persistence)` instead of the original project `w(p)=p`.
   In H0 every birth is zero, so the paper's `10*(tanh(p)+log(100*b+1))`
   reduces to `10*tanh(p)`.
3. **Gaussian bandwidth:** per layer, take the 0.2 quantile of the nonzero
   pairwise distances among the first 200 `(birth, persistence)` points from
   Teacher training diagrams. With H0 this is `abs(p_i-p_j)`.
4. **Global maximum:** compute the max of the unnormalized Teacher PIs across
   calibration batches **per layer**, then divide both Teacher and Student
   PIs by that same frozen maximum; never divide by the current batch max.
5. **Data-driven range:** use the minimum and maximum persistence values across
   all the Teacher calibration diagrams **for each layer**, rather than `[0,2]`.
   The bounds are frozen for both Teacher and Student for the full training run.
   In H0 the birth coordinate is always zero, so only the persistence axis needs
   minimum/maximum bounds. If the Teacher calibration deaths are all identical,
   use a minimal positive fallback interval so its width is never zero.

This follows the original's data-driven range. The PI is still represented in 1-D
with 100 point-sampled bins; the project still uses exact differentiable MST
H0, not the authors' 2-D 20x20 GUDHI PI or their RipsNet approximation.
Thus this is **not** an exact reproduction of the paper.

## Calibration and comparison

`fit_teacher_pi_calibrations` uses **only `loaders.train_eval`**, with test-time
transformations, no validation/test samples, no student outputs, and no
backpropagation. It uses the first `--pi-calibration-batches` batches (64 by
default). The original generates many teacher training PIs and uses one global
max; our calibration estimates that quantity from a bounded subset of teacher
training images, separately for each of the three layer pairs.

Run the paper-inspired variant in a *new* results directory, because the
results grid keys do not include PI mode and would otherwise silently skip
previous `KD+Top` experiments:

```bash
uv run python -m topkd_imat.training.run \
  --condition KD+Top --supervision label-supervised --init pretrained \
  --epochs 60 --batch-size 64 --w-top 0.1 \
  --pi-mode paper-h0 --pi-calibration-batches 64 \
  --runs-dir runs_paper_pi
```

The `config.json` includes the calibrated parameters so runs can be inspected.
Always retrain a plain KD control under identical seeds and training settings;
the `--pi-mode` option has no effect for conditions without Top.

**Important:** removing per-cloud scale normalization can put Student
persistences outside Teacher-calibrated ranges, leading to weak PI gradients.
Monitor gradient magnitudes and PI coverage. The coefficient `w_top=0.1`
was tuned on a different PI scale and **must be re-tuned**; comparing raw
loss numbers before/after normalization would be misleading.
