"""One training run: one Condition x Supervision x Initialisation x seed.

Loss = CE (label-supervised) + w_kd * KD (KD, KD+Top) + w_top * topology_loss (Top, KD+Top)
at the Layer Pairs ``top_pairs`` selects: by default the last one only (ADR 0007), or the mean
over all of them. The Teacher is frozen. Every term is logged for every Condition, trained on
or not, so a KD control's topology can be compared with KD+Top's: ``top`` is the term over the
selected Layer Pairs and ``top_<i>`` the term at Layer Pair i, selected or not (NaN when no
loss is given).
After the last epoch the Student is evaluated on the test set (through a Probe Head for
Top x label-free), its test-set clouds and checkpoint are saved, the topology term on the
test set goes to ``test_topology.json``, and one row is appended to the Results table.
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
    top_pairs: str = "last"  # "last" (ADR 0007) or "all" (mean over every Layer Pair)
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

    def selected_pairs(self, n_pairs: int) -> list[int]:
        """The Layer Pairs whose topology term is trained on (and reported as ``top``)."""
        if self.top_pairs == "last":
            return [n_pairs - 1]
        if self.top_pairs == "all":
            return list(range(n_pairs))
        raise ValueError(f"top_pairs must be 'last' or 'all', got {self.top_pairs!r}")

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
    if topology_loss is not None:
        with torch.no_grad():
            teacher_clouds, _ = collect(teacher, loaders.test, device)
        test_top = topology_between(
            topology_loss, teacher_clouds, clouds, cfg.batch_size, cfg.selected_pairs(n_pairs)
        )
        (run_dir / "test_topology.json").write_text(json.dumps(test_top, indent=2))
        print(f"[{run_dir.name}] test topology {test_top['top']:.4f}", flush=True)
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
    sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0}
    top_sums: np.ndarray | None = None
    correct = seen = top_seen = 0
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
        pairs = None
        if topology_loss is not None and x.shape[0] >= 2:
            selected = cfg.selected_pairs(len(t_feats))
            # With gradient only where it is trained on; the other pairs are only logged.
            pairs = _pair_terms(topology_loss, t_feats, s_feats, selected if cfg.uses_top else [])
            if cfg.uses_top:
                terms["top"] = torch.stack([pairs[i] for i in selected]).mean()
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
        for k in ("ce", "kd"):
            sums[k] += float(terms[k].detach()) * n
        if pairs is not None:
            values = np.array([float(v.detach()) for v in pairs]) * n
            top_sums = values if top_sums is None else top_sums + values
            top_seen += n
        correct += int((s_logits.argmax(1) == y).sum())
        seen += n
    out = {k: v / max(seen, 1) for k, v in sums.items()}
    selected = cfg.selected_pairs(len(top_sums)) if top_sums is not None else []
    out.update(_top_columns(top_sums, top_seen, selected))
    out["acc"] = correct / max(seen, 1)
    return out


def _pair_terms(topology_loss, t_feats, s_feats, with_grad=()) -> list[torch.Tensor]:
    """The topology term at each Layer Pair (one loss for all pairs, or one per pair).

    Only the pairs in ``with_grad`` keep a gradient path to the Student; the others are
    computed without gradient, for the logs.
    """
    if isinstance(topology_loss, (list, tuple)):
        layer_losses = topology_loss
    else:
        layer_losses = [topology_loss] * len(t_feats)
    if len(layer_losses) != len(t_feats):
        raise ValueError("Expected one topology loss per Teacher/Student layer pair")
    terms = []
    for i, (criterion, t, s) in enumerate(zip(layer_losses, t_feats, s_feats, strict=True)):
        if i in with_grad:
            terms.append(criterion(t, s))
        else:
            with torch.no_grad():
                terms.append(criterion(t, s.detach()))
    return terms


def _top_columns(top_sums: np.ndarray | None, seen: int, selected: list[int]) -> dict:
    """``top`` (over the selected Layer Pairs) and ``top_<i>``; NaN when nothing was measured."""
    if top_sums is None or not seen:
        return {"top": float("nan")}
    per_pair = [float(v) / seen for v in top_sums]
    top = float(np.mean([per_pair[i] for i in selected]))
    return {"top": top, **{f"top_{i}": v for i, v in enumerate(per_pair)}}


def topology_between(
    topology_loss, teacher_clouds, student_clouds, batch_size: int, selected=None
) -> dict:
    """The topology term between test-set clouds, in consecutive batches of ``batch_size``.

    The loss compares clouds of one batch, as in training, so the test set is cut into the
    same batch size (the incomplete last batch is dropped) and the terms are averaged.
    ``top`` is the mean over the ``selected`` Layer Pairs (default: all of them).
    """
    n = teacher_clouds[0].shape[0]
    top_sums, seen = None, 0
    with torch.no_grad():
        for start in range(0, n - batch_size + 1, batch_size):
            batch = slice(start, start + batch_size)
            pairs = _pair_terms(
                topology_loss,
                [c[batch].detach().double() for c in teacher_clouds],
                [c[batch].detach().double() for c in student_clouds],
            )
            values = np.array([float(v) for v in pairs])
            top_sums = values if top_sums is None else top_sums + values
            seen += 1
    selected = list(range(len(teacher_clouds))) if selected is None else list(selected)
    return {
        **_top_columns(top_sums, seen, selected),
        "selected_pairs": selected,
        "batches": seen,
        "batch_size": batch_size,
    }


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
