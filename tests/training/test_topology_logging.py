"""The topology term is logged for every Condition, per Layer Pair, and on the test set."""

import csv
import json
import math

import pytest
import torch

from tests.contracts.tinynet import TinyNet
from topkd_imat.topology.paper_pi import PaperH0PILoss, PaperPICalibration
from topkd_imat.training import run
from topkd_imat.training.data import make_loaders
from topkd_imat.training.loop import RunConfig, _pair_terms, topology_between, train_run

NORM = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
SPEC = PaperPICalibration(sigma=0.05, pi_max=5.0, bins=200, n_clouds=1)


@pytest.fixture
def checkpoints(tmp_path):
    torch.manual_seed(0)
    torch.save(TinyNet(width=8), tmp_path / "teacher.pt")
    torch.save(TinyNet(width=4), tmp_path / "student.pt")
    return tmp_path


def _metrics(run_dir):
    with (run_dir / "metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _run(checkpoints, condition, *extra):
    runs_dir = checkpoints / f"runs_{condition}_{'_'.join(extra)}"
    run.main(
        [
            "--condition", condition, "--supervision", "label-supervised", "--init", "scratch",
            "--dataset", "fake", "--fake-size", "256", "--num-workers", "0", "--device", "cpu",
            "--epochs", "2", "--runs-dir", str(runs_dir), "--pi-calibration-batches", "2",
            "--teacher-checkpoint", str(checkpoints / "teacher.pt"),
            "--student-checkpoint", str(checkpoints / "student.pt"), *extra,
        ]
    )  # fmt: skip
    (run_dir,) = (p for p in runs_dir.iterdir() if p.name != "teacher" and p.is_dir())
    return run_dir


@pytest.mark.parametrize("condition", ["KD", "KD+Top", "Baseline"])
def test_every_condition_logs_the_topology_term_per_layer_pair(checkpoints, condition):
    run_dir = _run(checkpoints, condition)
    for row in _metrics(run_dir):
        for split in ("train", "val"):
            pairs = [float(row[f"{split}_top_{i}"]) for i in range(2)]  # TinyNet: 2 pairs
            assert all(v > 0 and math.isfinite(v) for v in pairs)
            # By default only the last Layer Pair is the topology term (ADR 0007).
            assert float(row[f"{split}_top"]) == pytest.approx(pairs[-1], rel=1e-4)
    test = json.loads((run_dir / "test_topology.json").read_text())
    assert test["top"] == pytest.approx(test["top_1"]) and test["selected_pairs"] == [1]
    config = json.loads((run_dir / "config.json").read_text())
    assert (config["top_loss"], config["top_pairs"]) == ("h0-pi-norm", "last")


def test_top_pairs_all_averages_every_layer_pair(checkpoints):
    run_dir = _run(checkpoints, "KD+Top", "--top-pairs", "all")
    row = _metrics(run_dir)[-1]
    pairs = [float(row[f"val_top_{i}"]) for i in range(2)]
    assert float(row["val_top"]) == pytest.approx(sum(pairs) / 2, rel=1e-4)
    assert json.loads((run_dir / "test_topology.json").read_text())["selected_pairs"] == [0, 1]


@pytest.mark.parametrize(("top_pairs", "trained"), [("last", [1]), ("all", [0, 1])])
def test_only_the_selected_layer_pairs_get_a_gradient(top_pairs, trained):
    torch.manual_seed(0)
    teacher = [torch.randn(16, 4, dtype=torch.float64) for _ in range(2)]
    student = [torch.randn(16, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    selected = RunConfig(top_pairs=top_pairs).selected_pairs(2)
    assert selected == trained
    terms = _pair_terms(PaperH0PILoss(SPEC), teacher, student, selected)
    torch.stack([terms[i] for i in selected]).mean().backward()
    for i, cloud in enumerate(student):
        reached = cloud.grad is not None and bool(cloud.grad.abs().sum() > 0)
        assert reached == (i in trained)
        assert terms[i].requires_grad == (i in trained)


def _kd_run(tmp_path, topology_loss):
    torch.manual_seed(0)
    teacher, student = TinyNet(width=8), TinyNet(width=4)
    loaders = make_loaders("fake", NORM, batch_size=16, num_workers=0, fake_size=64)
    cfg = RunConfig(
        dataset="fake", condition="KD", init="scratch", epochs=2, runs_dir=str(tmp_path)
    )
    train_run(cfg, teacher, student, loaders, topology_loss)
    return student, _metrics(cfg.run_dir)


def test_logging_the_term_does_not_change_a_kd_run(tmp_path):
    plain, plain_rows = _kd_run(tmp_path / "plain", None)
    logged, logged_rows = _kd_run(tmp_path / "logged", PaperH0PILoss(SPEC))
    for k, v in plain.state_dict().items():
        torch.testing.assert_close(logged.state_dict()[k], v, rtol=0, atol=0)
    for a, b in zip(plain_rows, logged_rows, strict=True):
        assert (a["train_loss"], a["val_acc"]) == (b["train_loss"], b["val_acc"])
    # Without a loss nothing is measured: NaN, never a 0 that looks like a perfect match.
    assert math.isnan(float(plain_rows[-1]["val_top"]))
    assert float(logged_rows[-1]["val_top"]) > 0
    assert not list((tmp_path / "plain").glob("*/test_topology.json"))
    assert list((tmp_path / "logged").glob("*/test_topology.json"))


def test_test_topology_averages_whole_batches_per_layer_pair():
    torch.manual_seed(1)
    teacher = [torch.randn(70, 4), torch.randn(70, 6)]
    student = [torch.randn(70, 3), torch.randn(70, 5)]
    loss = PaperH0PILoss(SPEC)
    out = topology_between(loss, teacher, student, batch_size=32)
    assert out["batches"] == 2  # the last 6 points are dropped, as an incomplete batch
    for i in range(2):
        expected = (
            sum(
                loss(teacher[i][s : s + 32].double(), student[i][s : s + 32].double()).item()
                for s in (0, 32)
            )
            / 2
        )
        assert out[f"top_{i}"] == pytest.approx(expected)
    assert out["top"] == pytest.approx((out["top_0"] + out["top_1"]) / 2)
