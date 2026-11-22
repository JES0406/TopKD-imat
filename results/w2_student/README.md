# Student checks (W2)

Per-epoch logs behind the Student measurements in
[ADR 0003](../../docs/adr/0003-resnet56-teacher-pruned-resnet8-student.md). All runs: ResNet-8,
seed 0, 30 epochs, one RTX 2060 GPU. These are sanity checks for choosing the Student, not
grid results; the definitive Baseline numbers come from `topkd_imat.training.grid`.

| File | What | Final test accuracy |
|---|---|---|
| `cifar100_heal_from_pruned.csv` | Pruned ResNet-8 (from chenyaofo's CIFAR-100 ResNet-20) retrained on CIFAR-100: the published *pretrained* weights | 57.45 % (CIFAR-100) |
| `cifar100_heal_from_random.csv` | Same recipe from random weights (reference) | 52.88 % (CIFAR-100) |
| `cifar10_baseline_pretrained_healed.csv` | CIFAR-10 Baseline (CE only) from the *pretrained* Student | 85.63 % |
| `cifar10_baseline_pretrained_not_healed.csv` | CIFAR-10 Baseline from the pruned, not retrained Student | 83.52 % |
| `cifar10_baseline_scratch.csv` | CIFAR-10 Baseline from random weights (*scratch*) | 83.17 % |

Columns:

- `cifar100_*`: `epoch, lr, train_loss, train_acc, test_acc, seconds`. `test_acc` is on the CIFAR-100 test set. Made by `scripts/heal_student_cifar100.py` (SGD lr 0.05 cosine, batch 128).
- `cifar10_*`: `epoch, lr, train_loss, train_acc, val_acc, seconds`. `val_acc` is on the training code's fixed 5k validation split. The last row, `test`, holds the CIFAR-10 test accuracy. Made by `scripts/train_student_baseline.py` (the training code's recipe: SGD, lr 0.01 *pretrained* / 0.1 *scratch*, ÷10 at 62.5/75/87.5 % of the epochs, batch 64).

The healed *pretrained* Student is ahead from the first epoch (73 % validation accuracy against 53 % and 47 %): a possible figure for "pretraining helps the Student learn faster".
