"""Train one run from the command line.

    uv run python -m topkd_imat.training.run --condition KD --supervision label-free \\
        --init pretrained --epochs 30

Models: if a checkpoint is given (--teacher-checkpoint / --student-checkpoint, defaulting to
TEACHER_CHECKPOINT / STUDENT_CHECKPOINT in ``topkd_imat.models``), that file is loaded;
otherwise the model is built with ``topkd_imat.models.build_teacher()`` / ``build_student()``
(deterministic, weights downloaded once). Each run's config.json records which. Everything
lands in ``<runs-dir>/``; a run whose row already exists in ``<runs-dir>/results.csv`` is
skipped.

Topology Distillation Loss (``--top-loss``, Top and KD+Top only): ``h0-pi`` (ADR 0004, the
default), ``h0-matched`` (ADR 0005, with correspondence) or ``h0-paper-pi`` (ADR 0006, the
PI calibrated on the Teacher's training clouds as in the paper's code). Run names do not
include the loss, so any loss but the default needs its own ``--runs-dir``.
"""

import argparse
import sys

import torch

from topkd_imat import models
from topkd_imat.topology import DEFAULT_LOSS, LOSSES
from topkd_imat.training.checkpoints import load_model, reset_weights
from topkd_imat.training.data import make_loaders
from topkd_imat.training.loop import RunConfig, seed_everything, train_run
from topkd_imat.training.results import CONDITIONS, INITS, SUPERVISIONS, has_row

PAPER_PI = "h0-paper-pi"  # needs the Teacher and the data to calibrate, so built here
TOP_LOSSES = (*LOSSES, PAPER_PI)


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
    p.add_argument(
        "--top-loss",
        choices=TOP_LOSSES,
        default=DEFAULT_LOSS,
        help="h0-pi (ADR 0004), h0-matched (ADR 0005) or h0-paper-pi (ADR 0006)",
    )
    p.add_argument(
        "--pi-calibration-batches",
        type=int,
        default=64,
        help="h0-paper-pi only: Teacher training batches used to calibrate the PIs",
    )
    p.add_argument("--runs-dir", default=d.runs_dir)
    p.add_argument("--data-root", default="data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--teacher-checkpoint", default=models.TEACHER_CHECKPOINT)
    p.add_argument("--student-checkpoint", default=models.STUDENT_CHECKPOINT)
    p.add_argument("--fake-size", type=int, default=256, help=argparse.SUPPRESS)
    return p


def get_model(spec: str | None, role: str, device: str | torch.device = "cpu"):
    """Return ``(model, source)``: the checkpoint ``spec`` if given, else the built model.

    ``source`` says where the model came from, for the run's config.json.
    """
    if spec:
        return load_model(spec, role, device), {"checkpoint": spec}
    if role == "teacher":
        return models.build_teacher(progress=False).to(device), {"built": "build_teacher()"}
    from topkd_imat.models.build import HEALED_STUDENT

    source = {"built": "build_student()", "healed_weights": HEALED_STUDENT}
    return models.build_student(progress=False).to(device), source


def build_topology_loss(name: str, teacher, loaders, device, calibration_batches: int):
    """Return ``(loss, calibration)``: one loss for every Layer Pair, or one per pair.

    ``h0-paper-pi`` is calibrated on the Teacher's training clouds, one calibration per
    Layer Pair, so it is a list of losses; ``calibration`` records them for config.json.
    """
    if name != PAPER_PI:
        from topkd_imat.topology import make_topology_loss

        return make_topology_loss(name), None
    from topkd_imat.topology.paper_pi import PaperH0PILoss, fit_teacher_pi_calibrations

    calibrated = fit_teacher_pi_calibrations(
        teacher.to(device), loaders.train_eval, device, max_batches=calibration_batches
    )
    return [PaperH0PILoss(c) for c in calibrated], [c.to_dict() for c in calibrated]


def main(argv: list[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)
    if args.pi_calibration_batches <= 0:
        raise ValueError("--pi-calibration-batches must be positive")
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
    if cfg.uses_top and args.top_loss != DEFAULT_LOSS and args.runs_dir == RunConfig().runs_dir:
        sys.exit(
            f"--top-loss {args.top_loss} needs its own --runs-dir (e.g. runs_{args.top_loss}): "
            "run names do not include the loss, so existing runs would be skipped."
        )
    if has_row(f"{cfg.runs_dir}/results.csv", cfg.key()):
        print(f"[{cfg.run_dir.name}] already in {cfg.runs_dir}/results.csv, skipping.")
        return None

    seed_everything(cfg.seed)
    teacher, teacher_source = get_model(args.teacher_checkpoint, "teacher", args.device)
    student, student_source = get_model(args.student_checkpoint, "student", args.device)
    if cfg.init == "scratch":
        reset_weights(student)

    loaders = make_loaders(
        cfg.dataset,
        models.INPUT_NORMALIZATION,
        root=args.data_root,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        fake_size=args.fake_size,
    )
    topology_loss, calibration = None, None
    if cfg.uses_top:
        topology_loss, calibration = build_topology_loss(
            args.top_loss, teacher, loaders, args.device, args.pi_calibration_batches
        )
    first = topology_loss[0] if isinstance(topology_loss, list) else topology_loss
    extra = {
        "teacher_source": teacher_source,
        "student_source": student_source,
        "input_normalization": models.INPUT_NORMALIZATION,
        "top_loss": args.top_loss if cfg.uses_top else None,
        "topology_loss": type(first).__name__ if first is not None else None,
        "pi_calibration": calibration,
        "pi_calibration_batches": args.pi_calibration_batches if calibration else None,
    }
    return train_run(cfg, teacher, student, loaders, topology_loss, args.device, extra)


if __name__ == "__main__":
    main()
