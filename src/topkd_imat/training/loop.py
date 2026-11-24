"""One training run: one Condition x Supervision x Initialisation x seed.

Loss = CE (label-supervised) + w_kd * KD (KD, KD+Top) + w_top * mean over Layer Pairs of
topology_loss (Top, KD+Top). The Teacher is frozen. After the last epoch the Student is
evaluated on the test set (through a Probe Head for Top x label-free), its test-set clouds
and checkpoint are saved, and one row is appended to the Results table.
"""

import csv
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from topkd_imat.training.contracts import check_model_pair
from topkd_imat.training.data import Loaders
from topkd_imat.training.kd import distill_kl
from topkd_imat.training.probe import collect, fit_probe
from topkd_imat.training.results import (
    append_row,
    run_name,
    uses_probe_head,
    validate_row,
)

LR_DECAY_AT = (0.625, 0.75, 0.875)  # fractions of the epochs (150/180/210 of 240 in TopKD)


@dataclass
class RunConfig:
    dataset: str = "CIFAR-10"
    condition: str = "KD"
    supervision: str = "label-supervised"
    init: str = "pretrained"
    seed: int = 0
    epochs: int = 30
    lr: float | None = None  # None: 0.1 from scratch, 0.01 to finetune a pretrained Student
    batch_size: int = 64
    momentum: float = 0.9
    weight_decay: float = 5e-4
    kd_T: float = 4.0
    w_ce: float = 1.0
    w_kd: float = 1.0
    w_top: float = 1.0
    runs_dir: str = "runs"

    @property
    def uses_ce(self) -> bool:
        return self.supervision == "label-supervised"

    @property
    def uses_kd(self) -> bool:
        return self.condition in ("KD", "KD+Top")

    @property
    def uses_top(self) -> bool:
        return self.condition in ("Top", "KD+Top")

    def key(self) -> dict:
        return {
            "dataset": self.dataset,
            "condition": self.condition,
            "supervision": self.supervision,
            "init": self.init,
            "seed": self.seed,
        }

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / run_name(self.key())

    @property
    def effective_lr(self) -> float:
        if self.lr is not None:
            return self.lr
        return 0.1 if self.init == "scratch" else 0.01


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_run(
    cfg: RunConfig,
    teacher: nn.Module,
    student: nn.Module,
    loaders: Loaders,
    topology_loss=None,
    device: str | torch.device = "cpu",
    extra_config: dict | None = None,
) -> dict:
    """Train ``student`` in place and record the run; return its Results row."""
    validate_row({**cfg.key(), "accuracy": 0.0, "topology_distance": float("nan")})
    if cfg.uses_top and topology_loss is None:
        raise ValueError(f"Condition {cfg.condition} needs a topology_loss.")
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher, student = teacher.to(device).eval(), student.to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    x0, _ = next(iter(loaders.val))
    with torch.no_grad():
        n_pairs = check_model_pair(teacher, student, x0[:4].to(device), loaders.num_classes)

    _write_config(cfg, run_dir, n_pairs, extra_config or {})
    _save_teacher_clouds(teacher, loaders, device, Path(cfg.runs_dir))

    opt = torch.optim.SGD(
        student.parameters(),
        lr=cfg.effective_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    milestones = sorted({max(1, int(f * cfg.epochs)) for f in LR_DECAY_AT})
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    metrics_path = run_dir / "metrics.csv"
    metrics_path.unlink(missing_ok=True)
    try:
        for epoch in range(1, cfg.epochs + 1):
            start = time.time()
            lr = opt.param_groups[0]["lr"]
            train = _epoch(cfg, teacher, student, loaders.train, topology_loss, device, opt)
            sched.step()
            with torch.no_grad():
                val = _epoch(cfg, teacher, student, loaders.val, topology_loss, device, None)
            row = {"epoch": epoch, "lr": lr, "seconds": round(time.time() - start, 2)}
            row.update({f"train_{k}": v for k, v in train.items()})
            row.update({f"val_{k}": v for k, v in val.items()})
            _append_metrics(metrics_path, row)
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k.replace("_", "/", 1), v, epoch)
            print(
                f"[{run_dir.name}] epoch {epoch}/{cfg.epochs} "
                f"loss {train['loss']:.4f} val_acc {val['acc']:.4f} ({row['seconds']}s)",
                flush=True,
            )
    finally:
        writer.close()

    accuracy, clouds, labels = _evaluate(cfg, student, loaders, device)
    np.savez_compressed(
        run_dir / "clouds_test.npz",
        labels=labels.numpy(),
        **{f"pair_{i}": c.numpy().astype(np.float16) for i, c in enumerate(clouds)},
    )
    torch.save(student.cpu(), run_dir / "student.pt")
    result = {**cfg.key(), "accuracy": accuracy, "topology_distance": float("nan")}
    append_row(Path(cfg.runs_dir) / "results.csv", result)
    print(f"[{run_dir.name}] test accuracy {accuracy:.4f}", flush=True)
    return result


def _epoch(cfg, teacher, student, loader, topology_loss, device, opt) -> dict:
    """One pass; trains if ``opt`` is given. Returns mean loss terms and head accuracy."""
    student.train(opt is not None)
    sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "top": 0.0}
    correct = seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            t_logits, t_feats = teacher(x)
        s_logits, s_feats = student(x)
        terms = {}
        if cfg.uses_ce:
            terms["ce"] = F.cross_entropy(s_logits, y)
        else:
            with torch.no_grad():
                terms["ce"] = F.cross_entropy(s_logits, y)  # logged, not trained on
        if cfg.uses_kd:
            terms["kd"] = distill_kl(s_logits, t_logits, cfg.kd_T)
        else:
            with torch.no_grad():
                terms["kd"] = distill_kl(s_logits, t_logits, cfg.kd_T)  # logged, not trained on
        if cfg.uses_top and x.shape[0] >= 2:
            layer_losses = (
                topology_loss
                if isinstance(topology_loss, (list, tuple))
                else [topology_loss] * len(t_feats)
            )
            if len(layer_losses) != len(t_feats):
                raise ValueError("Expected one topology loss per Teacher/Student layer pair")
            pairs = [
                criterion(t, s)
                for criterion, t, s in zip(layer_losses, t_feats, s_feats, strict=True)
            ]
            terms["top"] = torch.stack(pairs).mean()
        loss = (
            (cfg.w_ce * terms["ce"] if cfg.uses_ce else 0.0)
            + (cfg.w_kd * terms["kd"] if cfg.uses_kd else 0.0)
            + (cfg.w_top * terms["top"] if "top" in terms else 0.0)
        )
        if opt is not None:
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                raise RuntimeError(f"{cfg.condition} x {cfg.supervision} has nothing to train on.")
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Training loss became {loss.item()}; lower the learning rate or w_top."
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        n = x.shape[0]
        sums["loss"] += float(torch.as_tensor(loss).detach()) * n
        for k, v in terms.items():
            sums[k] += float(v.detach()) * n
        correct += int((s_logits.argmax(1) == y).sum())
        seen += n
    out = {k: v / max(seen, 1) for k, v in sums.items()}
    out["acc"] = correct / max(seen, 1)
    return out


def _evaluate(cfg, student, loaders, device):
    """Test accuracy (head or Probe Head) and the test-set clouds."""
    clouds, labels = collect(student, loaders.test, device)
    if uses_probe_head(cfg.condition, cfg.supervision):
        train_feats, train_labels = collect(student, loaders.train_eval, device)
        probe = fit_probe(train_feats[-1], train_labels, loaders.num_classes)
        with torch.no_grad():
            pred = probe(clouds[-1]).argmax(1)
    else:
        student.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loaders.test:
                preds.append(student(x.to(device))[0].argmax(1).cpu())
        pred = torch.cat(preds)
    return float((pred == labels).float().mean()), clouds, labels


def _save_teacher_clouds(teacher, loaders, device, runs_dir: Path) -> None:
    path = runs_dir / "teacher" / "clouds_test.npz"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    clouds, labels = collect(teacher, loaders.test, device)
    np.savez_compressed(
        path,
        labels=labels.numpy(),
        **{f"pair_{i}": c.numpy().astype(np.float16) for i, c in enumerate(clouds)},
    )


def _append_metrics(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _write_config(cfg: RunConfig, run_dir: Path, n_pairs: int, extra: dict) -> None:
    config = {
        **asdict(cfg),
        "effective_lr": cfg.effective_lr,
        "layer_pairs": n_pairs,
        "torch": torch.__version__,
        **_git_state(),
        **extra,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
