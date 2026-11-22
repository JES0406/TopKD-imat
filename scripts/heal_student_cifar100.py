"""How the pretrained Student's weights were made: prune the CIFAR-100 ResNet-20 to ResNet-8,
then retrain it on CIFAR-100 ("healing", standard after pruning) so it works again.

    uv run python scripts/heal_student_cifar100.py --epochs 30

The file this produced (seed 0, RTX 2060, 57.45 % CIFAR-100 test) is published on the Hugging
Face Hub and pinned in topkd_imat/models/build.py (HEALED_STUDENT); every run uses that file.
Re-running gives an equally good but not bit-identical one (floating-point order differs
across hardware), so do not replace the published file without telling the team.

Keeps the original 100-class head during healing; the 10-class head for CIFAR-10 is attached
later by build_student. Uses CIFAR-100 only, so the Student still never sees CIFAR-10 labels.
Writes the healed state_dict to checkpoints/ and one CSV row per epoch to runs/student_heal/.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from topkd_imat.models.build import UPSTREAM_NORMALIZATION, load_upstream
from topkd_imat.models.resnet_cifar import prune_depth
from topkd_imat.training.checkpoints import reset_weights
from topkd_imat.training.loop import seed_everything

HEALED_WEIGHTS = Path("checkpoints/student_resnet8_healed_c100_state_dict.pt")


def loaders(batch_size: int, num_workers: int, root: str = "data"):
    norm = transforms.Normalize(*UPSTREAM_NORMALIZATION["cifar100"])
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR100(root, train=True, transform=train_tf, download=True)
    test = datasets.CIFAR100(root, train=False, transform=test_tf, download=True)
    return (
        DataLoader(
            train,
            batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
        DataLoader(test, 500, num_workers=num_workers, persistent_workers=num_workers > 0),
    )


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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--scratch",
        action="store_true",
        help="reference run: same recipe from random weights (is the online start worth it?)",
    )
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    seed_everything(args.seed)
    model = prune_depth(load_upstream("cifar100", 20, progress=False), 1)
    if args.scratch:
        reset_weights(model)
    model = model.to(args.device)
    train, test = loaders(args.batch_size, args.num_workers)
    before = evaluate(model, test, args.device)
    print(f"pruned ResNet-8 before healing: CIFAR-100 test {before:.4f}", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path("runs/student_heal")
    out.mkdir(parents=True, exist_ok=True)
    tag = "scratch" if args.scratch else "heal"
    log = out / f"{tag}_lr{args.lr}_e{args.epochs}_s{args.seed}.csv"
    with log.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "test_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            loss_sum = correct = seen = 0
            for x, y in train:
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
            lr = opt.param_groups[0]["lr"]
            sched.step()
            acc = evaluate(model, test, args.device)
            secs = time.time() - start
            writer.writerow([epoch, lr, loss_sum / seen, correct / seen, acc, round(secs, 1)])
            f.flush()
            print(
                f"[heal] epoch {epoch}/{args.epochs} loss {loss_sum / seen:.4f} "
                f"train_acc {correct / seen:.4f} c100_test {acc:.4f} ({secs:.0f}s)",
                flush=True,
            )
    if args.scratch:
        print(f"scratch ResNet-8 reference: CIFAR-100 test {acc:.4f}", flush=True)
        return
    HEALED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), HEALED_WEIGHTS)
    print(f"healed ResNet-8: CIFAR-100 test {acc:.4f} -> {HEALED_WEIGHTS}", flush=True)


if __name__ == "__main__":
    main()
