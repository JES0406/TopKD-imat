import json

import pytest
import torch

from tests.contracts.tinynet import TinyNet
from topkd_imat.training import run
from topkd_imat.training.run import TOP_LOSSES, build_parser

BASE = ["--condition", "Top", "--supervision", "label-free", "--init", "scratch"]


def test_top_loss_defaults_to_the_normalised_calibrated_pi():
    assert build_parser().parse_args(BASE).top_loss == "h0-pi-norm"


def test_top_loss_offers_exactly_the_three_losses():
    assert set(TOP_LOSSES) == {"h0-pi-norm", "h0-matched"}
    for name in TOP_LOSSES:
        assert build_parser().parse_args([*BASE, "--top-loss", name]).top_loss == name
    for gone in ("h0-pi", "h0-pi-raw", "h0-paper-pi"):  # no longer run options (ADR 0006)
        with pytest.raises(SystemExit):
            build_parser().parse_args([*BASE, "--top-loss", gone])


@pytest.fixture
def checkpoints(tmp_path):
    torch.manual_seed(0)
    torch.save(TinyNet(width=8), tmp_path / "teacher.pt")
    torch.save(TinyNet(width=4), tmp_path / "student.pt")
    return tmp_path


def _argv(ckpt, loss, runs_dir, condition="KD+Top"):
    return [
        "--condition", condition, "--supervision", "label-supervised", "--init", "pretrained",
        "--dataset", "fake", "--fake-size", "256", "--num-workers", "0", "--device", "cpu",
        "--epochs", "1", "--top-loss", loss, "--runs-dir", str(runs_dir),
        "--pi-calibration-batches", "2",
        "--teacher-checkpoint", str(ckpt / "teacher.pt"),
        "--student-checkpoint", str(ckpt / "student.pt"),
    ]  # fmt: skip


@pytest.mark.parametrize("loss", TOP_LOSSES)
def test_every_top_loss_trains_and_is_recorded(checkpoints, loss):
    runs_dir = checkpoints / f"runs_{loss}"
    row = run.main(_argv(checkpoints, loss, runs_dir))
    assert row is not None and 0.0 <= row["accuracy"] <= 1.0
    (config_path,) = runs_dir.glob("*/config.json")
    config = json.loads(config_path.read_text())
    assert config["top_loss"] == loss
    expected = {"h0-pi-norm": "PaperH0PILoss", "h0-matched": "H0MatchedLoss"}
    assert config["topology_loss"] == expected[loss]
    if loss == "h0-matched":
        assert config["pi_calibration"] is None
    else:
        assert len(config["pi_calibration"]) == 2  # one per TinyNet Layer Pair


@pytest.mark.parametrize("loss", TOP_LOSSES)
def test_every_top_loss_refuses_the_shared_runs_dir(checkpoints, monkeypatch, loss):
    # The default loss too: older runs/ folders hold ADR 0004's h0-pi under the same names.
    monkeypatch.chdir(checkpoints)
    with pytest.raises(SystemExit, match="needs its own --runs-dir"):
        run.main(_argv(checkpoints, loss, "runs"))


def test_runs_without_top_may_use_the_shared_runs_dir(checkpoints, monkeypatch):
    monkeypatch.chdir(checkpoints)
    row = run.main(_argv(checkpoints, "h0-pi-norm", "runs", condition="KD"))
    assert row is not None
