# ResNet-56 Teacher, ResNet-8 Student pruned from a CIFAR-100 ResNet-20

*Status: accepted by the team on 2026-11-22 (Teacher, Student architecture, Layer Pairs); the pretrained weights are still open, see below.*

The Teacher is the CIFAR-10 **ResNet-56** released by [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) (94.37 % test accuracy, reproduced exactly here; 0.86 M parameters). The Student is a **ResNet-8**: the CIFAR-100 ResNet-20 from the same repository, depth-pruned by keeping the first BasicBlock of each stage, with a fresh 10-class head (0.08 M parameters). The *pretrained* Student is that network as built; the *scratch* Student is the same ResNet-8 with its weights reset. There are **three Layer Pairs**, one per stage (global-average-pooled stage outputs, dims 16 / 32 / 64 in both networks), and the last one is the classifier's input.

ResNet-56 → ResNet-20 is the pair TopKD reports first (Kim et al., 2024, Table 1, from the CRD benchmark of Tian et al., 2020), but on CIFAR-10 a ResNet-20 alone already reaches 92.60 %, only ~1.8 points below the Teacher, while TopKD's gain in the paper is ~2.5 points on a 3.3-point gap (CIFAR-100; variance across runs ≈ 0.2, Table 6). With a gap that small the Conditions would be indistinguishable from seed noise. Pruning to ResNet-8 widens the gap to ~10 points (Baseline below: 84.2 % / 83.5 % vs 94.4 %), inside the range of TopKD's own pairs (2–15 points) while keeping the Student in the Teacher's family: same stem, same stages and widths, so Layer Pair *k* is stage *k* of both networks and no projection between latent dimensions is needed. Weights come from CIFAR-100 rather than CIFAR-10 so the *pretrained* Student has never seen CIFAR-10 labels. No public small-network weights fitted better: online CIFAR checkpoints are either trained on CIFAR-10 itself (no gap, leaks the task) or ImageNet models for 224×224 inputs.

Inputs are normalised with the Teacher's training statistics, mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010), read from chenyaofo's training log.

## Measurements (scripts/evaluate_models.py, scripts/train_student_baseline.py)

| | Test accuracy |
|---|---|
| Teacher, ResNet-56, CIFAR-10 | 94.37 % (94.25 % with the true CIFAR-10 std: normalisation mismatch costs little here) |
| Source ResNet-20 on CIFAR-100, as released | 68.83 % |
| … pruned to ResNet-8, no retraining | 1.29 % (5.77 % after recomputing BatchNorm statistics); ResNet-14 for comparison: 7.11 % / 23.68 % |
| Student as handed to training, CIFAR-10 | 10.24 % (fresh head: chance) |
| Student Baseline, *pretrained* (CE only, lr 0.01, 30 epochs, seed 0) | 84.20 % |
| Student Baseline, *scratch* (CE only, lr 0.1, 30 epochs, seed 0) | 83.52 % |

The two Baseline rows come from `scripts/train_student_baseline.py`, which uses the training code's data split, augmentation and recipe but skips the Teacher's forward pass so it runs on a laptop CPU; the definitive Baseline numbers are the grid's (W4).

## Alternative under discussion: heal the pruned ResNet-8 on CIFAR-100

Standard practice after pruning is to retrain the pruned network so its remaining blocks fit together again. `scripts/heal_student_cifar100.py` retrains the pruned ResNet-8 on CIFAR-100 (30 epochs, SGD lr 0.05 cosine, batch 128; ~4 min on a GPU) before the 10-class head is attached (`build_student(healed_weights=...)`). It still never sees CIFAR-10 labels. Measured on one seed, on a GPU:

| | CIFAR-100 test | CIFAR-10 Baseline (CE only, 30 epochs) |
|---|---|---|
| Pruned ResNet-8, healed (online start) | 57.45 % | **85.63 %** |
| Same recipe from random weights (reference) | 52.88 % | — |
| Pruned ResNet-8, not healed (current *pretrained*) | 1.29 % | 83.52 % |
| *Scratch* | — | 83.17 % |

The healed *pretrained* Student is a working network (the online start is worth +4.6 points on CIFAR-100 over random weights), learns CIFAR-10 much faster (73 % validation accuracy after one epoch vs 53 % / 47 %) and ends ~2 points above both other starts, so the *pretrained*/*scratch* axis would measure something. The gap to the Teacher stays ~9 points. The same seed gave 84.20 % / 83.52 % on a CPU for the non-healed / scratch runs, so differences under ~0.7 points are noise here. Cost: the healed weights come from our own training, which is seeded but not bit-identical across hardware, so one file must be shared (e.g. on the Hugging Face Hub) or built once on the training machine (the DGX) and used by every run there.

**Team decision (2026-11-22):** the healed Student is preferred — it is real pretraining — if it is wired in by Tuesday 2026-11-24; otherwise the grid runs with the current non-healed Student. Until then `build_student()` returns the non-healed one.

**Deviation from the plan.** The plan defines *pretrained* as "the Student starts from weights found online". Neither Student does exactly that: the non-healed one starts from online weights with two thirds of its blocks removed, and the healed one from those weights retrained by us on CIFAR-100. Report the *pretrained* Condition in the memoria as "initialised from public CIFAR-100 weights, pruned (and retrained on CIFAR-100)", not as downloaded weights.

## Consequences

- *Pretrained* keeps the plan's meaning (weights found online), but pruning to one block per stage destroys most of what the source network computed (5.8 % on its own CIFAR-100 test set even with recalibrated BatchNorm). The *pretrained* Student starts from partly useful early filters, not from a working network; the *scratch*/*pretrained* axis measures how much that start helps. In the Baseline check it helped early (79 % vs 61–71 % validation accuracy at epoch 12) but only by 0.7 points at the end (one seed, likely within noise). Train and validation accuracy stay close (86 % / 85 %), so the ResNet-8 is limited by capacity, not overfitting.
- The class lives in `topkd_imat.models.resnet_cifar` (adapted from chenyaofo's code, BSD 3-Clause, notices in `THIRD_PARTY_LICENSES.md`); its parameter names match the released state_dicts, which load with `strict=True`. `scripts/build_models.py` builds both whole-model checkpoints deterministically (the Student's new head, its only random part, is seeded), so each training machine builds them itself and nothing is uploaded; `TEACHER_CHECKPOINT` / `STUDENT_CHECKPOINT` stay `None` and runs pass the paths.
- H0 check of the Teacher's last Latent Point Cloud (Euclidean MST = Rips H0 death times) on a class-balanced batch of 100 test images: cutting the 9 longest bars leaves 10 components of 7–12 points with 95 % label purity, so the Class Clusters are there. But the 9 longest bars (6.05 … 4.13) are not separated from the 10th–12th (3.80, 3.73, 3.59) by a clear gap, only from the median bar (1.51); with 200 points the purity drops to 77 % (two classes merge, two outliers become their own components). "9 long bars" is a tendency to show with care in the figures, and it depends on the batch size (W3, W5).
- The Teacher's clouds are ~4× larger in scale than the untrained Student's (longest bars ~5 vs ~1.2), so scale normalisation of the clouds matters for the Topology Distillation Loss (W3).
