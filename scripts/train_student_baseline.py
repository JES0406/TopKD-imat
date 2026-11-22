"""How good is the Student on its own? Trains it with cross-entropy only (the Baseline
Condition) on CIFAR-10, without a Teacher, to measure the Teacher-Student gap cheaply.

    uv run python scripts/train_student_baseline.py --init pretrained --epochs 30
    uv run python scripts/train_student_baseline.py --init scratch --epochs 30

Same recipe as topkd_imat.training (SGD, momentum 0.9, weight decay 5e-4, batch 64, lr 0.1
scratch / 0.01 pretrained, x0.1 at 62.5/75/87.5 % of the epochs, the same data split and
augmentation), but it skips the Teacher's forward pass, so it is fast enough for a laptop.
The definitive Baseline numbers come from the real grid (topkd_imat.training.grid).
Writes one CSV per run to runs/student_baseline/.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from topkd_imat import models
from topkd_imat.models.build import build_student
from topkd_imat.training.checkpoints import reset_weights
from topkd_imat.training.data import make_loaders
from topkd_imat.training.loop import LR_DECAY_AT, seed_everything


@torch.no_grad()
def evaluate(model, loader, device="cpu") -> float:
    model.eval()
    correct = seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += int((model(x)[0].argmax(1) == y).sum())
        seen += len(y)
    return correct / seen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--init", choices=["scratch", "pretrained"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--healed-weights",
        default=None,
        help="pretrained = pruned ResNet-8 healed on CIFAR-100 (scripts/heal_student_cifar100.py)",
    )
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    seed_everything(args.seed)
    model = build_student(progress=False, healed_weights=args.healed_weights)
    if args.init == "scratch":
        reset_weights(model)
    model = model.to(args.device)
    lr = args.lr if args.lr is not None else (0.1 if args.init == "scratch" else 0.01)
    loaders = make_loaders("CIFAR-10", models.INPUT_NORMALIZATION, num_workers=args.num_workers)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    milestones = sorted({max(1, int(f * args.epochs)) for f in LR_DECAY_AT})
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    out = Path("runs/student_baseline")
    out.mkdir(parents=True, exist_ok=True)
    init = f"{args.init}-healed" if args.healed_weights else args.init
    path = out / f"{init}_lr{lr}_e{args.epochs}_s{args.seed}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            loss_sum = correct = seen = 0
            for x, y in loaders.train:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += float(loss) * len(y)
                correct += int((logits.argmax(1) == y).sum())
                seen += len(y)
            current_lr = opt.param_groups[0]["lr"]
            sched.step()
            val = evaluate(model, loaders.val, args.device)
            secs = time.time() - start
            writer.writerow(
                [epoch, current_lr, loss_sum / seen, correct / seen, val, round(secs, 1)]
            )
            f.flush()
            print(
                f"[{args.init}] epoch {epoch}/{args.epochs} loss {loss_sum / seen:.4f} "
                f"train_acc {correct / seen:.4f} val_acc {val:.4f} ({secs:.0f}s)",
                flush=True,
            )
        test = evaluate(model, loaders.test, args.device)
        writer.writerow(["test", "", "", "", test, ""])
    print(f"[{args.init}] TEST accuracy {test:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
