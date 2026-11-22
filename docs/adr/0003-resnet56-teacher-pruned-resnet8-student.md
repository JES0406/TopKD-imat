# ResNet-56 Teacher, ResNet-8 Student pruned from a CIFAR-100 ResNet-20 and retrained

*Status: accepted by the team on 2026-11-22.*

The Teacher is the CIFAR-10 **ResNet-56** released by [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) (94.37 % test accuracy, reproduced exactly here; 0.86 M parameters). The Student is a **ResNet-8** (0.08 M parameters): the CIFAR-100 ResNet-20 from the same repository, depth-pruned by keeping the first BasicBlock of each stage, **retrained ("healed") by us on CIFAR-100**, then given a fresh, seeded 10-class head. The *pretrained* Student is that network; the *scratch* Student is the same ResNet-8 with its weights reset. There are **three Layer Pairs**, one per stage (global-average-pooled stage outputs, dims 16 / 32 / 64 in both networks), and the last one is the classifier's input.

ResNet-56 → ResNet-20 is the pair TopKD reports first (Kim et al., 2024, Table 1, from the CRD benchmark of Tian et al., 2020), but on CIFAR-10 a ResNet-20 alone already reaches 92.60 %, only ~1.8 points below the Teacher, while TopKD's gain in the paper is ~2.5 points on a 3.3-point gap (CIFAR-100; variance across runs ≈ 0.2, Table 6). With a gap that small the Conditions would be indistinguishable from seed noise. A ResNet-8 widens the gap to ~9–11 points (Baseline below), inside the range of TopKD's own pairs (2–15 points), while keeping the Student in the Teacher's family: same stem, same stages and widths, so Layer Pair *k* is stage *k* of both networks and no projection between latent dimensions is needed. Weights come from CIFAR-100 rather than CIFAR-10 so the *pretrained* Student has never seen CIFAR-10 labels. No public small-network weights fitted better: online CIFAR checkpoints are either trained on CIFAR-10 itself (no gap, leaks the task) or ImageNet models for 224×224 inputs.

Pruning alone is not enough: removing two thirds of the blocks destroys what the source network computed (1.29 % on its own CIFAR-100 test set, 5.77 % even after recomputing the BatchNorm statistics). As is standard after pruning, the pruned ResNet-8 is retrained on CIFAR-100 (`scripts/heal_student_cifar100.py`: 30 epochs, SGD lr 0.05 cosine, batch 128, seed 0; ~4 min on a GPU). The result is a working network, and starting from the online weights is worth +4.6 points over the same recipe from random weights.

Inputs are normalised with the Teacher's training statistics, mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010), read from chenyaofo's training log.

## Measurements

Seed 0. Baselines: cross-entropy only, 30 epochs, the training code's split, augmentation and recipe (lr 0.01 *pretrained*, 0.1 *scratch*), run with `scripts/train_student_baseline.py`, which skips the Teacher's forward pass; the definitive Baseline numbers are the grid's (W4). Per-epoch logs of the GPU runs: [`results/w2_student/`](../../results/w2_student/).

| | CIFAR-100 test | CIFAR-10 test |
|---|---|---|
| Teacher, ResNet-56 | — | 94.37 % (94.25 % with the true CIFAR-10 std) |
| Source ResNet-20, as released | 68.83 % | — |
| … pruned to ResNet-8, no retraining | 1.29 % (5.77 % with BatchNorm recomputed; ResNet-14: 7.11 % / 23.68 %) | — |
| … pruned and healed (**the *pretrained* Student before CIFAR-10 training**) | **57.45 %** | 10 % (fresh head: chance) |
| ResNet-8 from random weights, same healing recipe (reference) | 52.88 % | — |
| **Baseline, *pretrained* (healed)** | — | **85.63 %** (73 % validation after one epoch) |
| Baseline, pruned but not healed | — | 83.52 % GPU / 84.20 % CPU (53 % after one epoch) |
| **Baseline, *scratch*** | — | **83.17 %** GPU / 83.52 % CPU (47 % after one epoch) |

The same seed on CPU and GPU differs by up to 0.7 points, so smaller differences are noise. The healed *pretrained* Student learns faster and ends ~2.5 points above *scratch*, so the *pretrained*/*scratch* axis measures something; the non-healed one ended within noise of *scratch*, which is why the healed one was chosen. Train and validation accuracy stay close (~86 % / 85 %): the ResNet-8 is limited by capacity, not overfitting.

## Consequences

- **Deviation from the plan.** The plan defines *pretrained* as "the Student starts from weights found online". The Student starts from online weights that we pruned and retrained on CIFAR-100. Report the *pretrained* Initialisation in the memoria as "initialised from public CIFAR-100 weights, pruned and retrained on CIFAR-100", not as downloaded weights.
- **Where the weights live.** The healed weights come from our own training, which is seeded but not bit-identical across hardware (floating-point summation order differs), so one file is shared: the state_dict produced on an RTX 2060 is on the Hugging Face Hub at [`MarioAlonso7/topkd`](https://huggingface.co/MarioAlonso7/topkd), and `topkd_imat.models.build.HEALED_STUDENT` pins its revision and sha256. `build_teacher()` / `build_student()` download what they need on first use and are deterministic, so every machine builds identical models; training builds them directly when no checkpoint path is given. `build_student(healed=False)` keeps the non-healed Student for comparison.
- The class lives in `topkd_imat.models.resnet_cifar` (adapted from chenyaofo's code, BSD 3-Clause, notices in `THIRD_PARTY_LICENSES.md`); its parameter names match the released state_dicts, which load with `strict=True`.
- H0 check of the Teacher's last Latent Point Cloud (Euclidean MST = Rips H0 death times) on a class-balanced batch of 100 test images: cutting the 9 longest bars leaves 10 components of 7–12 points with 95 % label purity, so the Class Clusters are there. But the 9 longest bars (6.05 … 4.13) are not separated from the 10th–12th (3.80, 3.73, 3.59) by a clear gap, only from the median bar (1.51); with 200 points the purity drops to 77 % (two classes merge, two outliers become their own components). "9 long bars" is a tendency to show with care in the figures, and it depends on the batch size (W3, W5).
- The Teacher's clouds are ~4× larger in scale than an untrained Student's (longest bars ~5 vs ~1.2), so scale normalisation of the clouds matters for the Topology Distillation Loss (W3).
