"""Probe Head reference points, to read the Top x label-free accuracy against.

    uv run python -m topkd_imat.training.probe_reference --runs-dir runs_final --trained

Fits the same Probe Head as the grid (linear, L-BFGS, last Layer Pair, training split with
test-time transforms) on models the grid did not train, and writes
``<runs-dir>/probe_reference.csv`` with one row per model:

- ``teacher``: the Teacher (the ceiling);
- ``student_pretrained_untrained`` and ``student_scratch_untrained_s<seed>``: the Student as
  each run starts, before any training. Each is probed twice: with the BatchNorm statistics
  it was built with (``bn=as-built``) and with them recomputed on the training split
  (``bn=recomputed``), since the built ones do not come from CIFAR-10;
- with ``--trained``: every ``student.pt`` saved in ``<runs-dir>``, so every Condition gets
  the same last-layer measure (``bn=trained``).

Not part of the Results table: the probe accuracy is a separate measure.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from topkd_imat import models
from topkd_imat.training.checkpoints import load_model, reset_weights
from topkd_imat.training.data import make_loaders
from topkd_imat.training.loop import seed_everything
from topkd_imat.training.probe import collect, fit_probe
from topkd_imat.training.run import get_model

FIELDS = ["model", "bn", "probe_accuracy"]


@torch.no_grad()
def recompute_batchnorm(model: nn.Module, loader, device) -> int:
    """Replace every BatchNorm's running statistics with their average over ``loader``.

    Returns how many BatchNorm layers were recomputed (0 leaves the model untouched).
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return 0
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the whole pass
    model.train()
    for x, _ in loader:
        model(x.to(device, non_blocking=True))
    for m, momentum in zip(norms, momenta, strict=True):
        m.momentum = momentum
    model.eval()
    return len(norms)


def probe_accuracy(model: nn.Module, loaders, device) -> float:
    """Test accuracy of a Probe Head fitted on ``model``'s last Layer Pair."""
    train_feats, train_labels = collect(model, loaders.train_eval, device)
    probe = fit_probe(train_feats[-1], train_labels, loaders.num_classes)
    test_feats, test_labels = collect(model, loaders.test, device)
    with torch.no_grad():
        pred = probe(test_feats[-1]).argmax(1)
    return float((pred == test_labels).double().mean())


def _untrained(args, init: str, seed: int, device):
    seed_everything(seed)  # same order as run.main, so this is the Student each run starts from
    get_model(args.teacher_checkpoint, "teacher", device)  # keeps the random stream aligned
    student, _ = get_model(args.student_checkpoint, "student", device)
    if init == "scratch":
        reset_weights(student)
    return student.to(device)


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs-dir", default="runs_final")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--trained", action="store_true", help="also probe every saved student.pt")
    p.add_argument("--dataset", default="CIFAR-10")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fake-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--teacher-checkpoint", default=models.TEACHER_CHECKPOINT)
    p.add_argument("--student-checkpoint", default=models.STUDENT_CHECKPOINT)
    args = p.parse_args(argv)

    loaders = make_loaders(
        args.dataset,
        models.INPUT_NORMALIZATION,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fake_size=args.fake_size,
    )
    rows: list[dict] = []

    def record(name: str, bn: str, model: nn.Module) -> None:
        row = {
            "model": name,
            "bn": bn,
            "probe_accuracy": probe_accuracy(model, loaders, args.device),
        }
        print(f"{name:55} {bn:10} {row['probe_accuracy']:.4f}", flush=True)
        rows.append(row)

    teacher, _ = get_model(args.teacher_checkpoint, "teacher", args.device)
    record("teacher", "trained", teacher.to(args.device))

    starts = [("pretrained", args.seeds[0], "student_pretrained_untrained")]
    starts += [("scratch", s, f"student_scratch_untrained_s{s}") for s in args.seeds]
    for init, seed, name in starts:
        student = _untrained(args, init, seed, args.device)
        record(name, "as-built", student)
        if recompute_batchnorm(student, loaders.train_eval, args.device):
            record(name, "recomputed", student)

    if args.trained:
        for path in sorted(Path(args.runs_dir).glob("*/student.pt")):
            record(path.parent.name, "trained", load_model(str(path), "student", args.device))

    out = Path(args.runs_dir) / "probe_reference.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return rows


if __name__ == "__main__":
    main()
