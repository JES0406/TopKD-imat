"""Train one run from the command line.

    uv run python -m topkd_imat.training.run --condition KD --supervision label-free \\
        --init pretrained --epochs 30

Models come from ``topkd_imat.models`` (TEACHER_CHECKPOINT / STUDENT_CHECKPOINT) unless
overridden. Everything lands in ``<runs-dir>/``; a run whose row already exists in
``<runs-dir>/results.csv`` is skipped.
"""

import argparse
import sys

import torch

from topkd_imat import models
from topkd_imat.training.checkpoints import load_model, reset_weights
from topkd_imat.training.data import make_loaders
from topkd_imat.training.loop import RunConfig, seed_everything, train_run
from topkd_imat.training.results import CONDITIONS, INITS, SUPERVISIONS, has_row


def build_parser() -> argparse.ArgumentParser:
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--supervision", choices=SUPERVISIONS, required=True)
    p.add_argument("--init", choices=INITS, required=True)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--dataset", default=d.dataset, help="CIFAR-10, or 'fake' for tests")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=None, help="default 0.1 scratch / 0.01 pretrained")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--kd-T", type=float, default=d.kd_T)
    p.add_argument("--w-kd", type=float, default=d.w_kd)
    p.add_argument("--w-top", type=float, default=d.w_top)
    p.add_argument("--runs-dir", default=d.runs_dir)
    p.add_argument("--data-root", default="data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--teacher-checkpoint", default=models.TEACHER_CHECKPOINT)
    p.add_argument("--student-checkpoint", default=models.STUDENT_CHECKPOINT)
    p.add_argument("--fake-size", type=int, default=256, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)
    cfg = RunConfig(
        dataset=args.dataset,
        condition=args.condition,
        supervision=args.supervision,
        init=args.init,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        kd_T=args.kd_T,
        w_kd=args.w_kd,
        w_top=args.w_top,
        runs_dir=args.runs_dir,
    )
    if cfg.condition == "Baseline" and cfg.supervision == "label-free":
        sys.exit("Baseline trains on labels only; use --supervision label-supervised.")
    if has_row(f"{cfg.runs_dir}/results.csv", cfg.key()):
        print(f"[{cfg.run_dir.name}] already in {cfg.runs_dir}/results.csv, skipping.")
        return None

    seed_everything(cfg.seed)
    teacher = load_model(args.teacher_checkpoint, "teacher", args.device)
    student = load_model(args.student_checkpoint, "student", args.device)
    if cfg.init == "scratch":
        reset_weights(student)
    topology_loss = None
    if cfg.uses_top:
        from topkd_imat.topology import topology_loss

    loaders = make_loaders(
        cfg.dataset,
        models.INPUT_NORMALIZATION,
        root=args.data_root,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        fake_size=args.fake_size,
    )
    extra = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "input_normalization": models.INPUT_NORMALIZATION,
        "topology_loss": type(topology_loss).__name__ if topology_loss is not None else None,
    }
    return train_run(cfg, teacher, student, loaders, topology_loss, args.device, extra)


if __name__ == "__main__":
    main()
