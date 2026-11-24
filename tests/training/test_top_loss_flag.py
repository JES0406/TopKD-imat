import json

import pytest
import torch

from tests.contracts.tinynet import TinyNet
from topkd_imat.topology import LOSSES
from topkd_imat.training import run
from topkd_imat.training.run import TOP_LOSSES, build_parser

BASE = ["--condition", "Top", "--supervision", "label-free", "--init", "scratch"]


def test_top_loss_defaults_to_the_persistence_image_loss():
    assert build_parser().parse_args(BASE).top_loss == "h0-pi"


def test_top_loss_accepts_every_registered_loss_and_the_paper_pi():
    assert set(TOP_LOSSES) == {*LOSSES, "h0-paper-pi"}
    for name in TOP_LOSSES:
        assert build_parser().parse_args([*BASE, "--top-loss", name]).top_loss == name


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
    expected = {"h0-pi": "H0PILoss", "h0-matched": "H0MatchedLoss", "h0-paper-pi": "PaperH0PILoss"}
    assert config["topology_loss"] == expected[loss]
    if loss == "h0-paper-pi":
        assert len(config["pi_calibration"]) == 2  # one per TinyNet Layer Pair
    else:
        assert config["pi_calibration"] is None


def test_a_non_default_loss_refuses_the_shared_runs_dir(checkpoints, monkeypatch):
    monkeypatch.chdir(checkpoints)
    argv = _argv(checkpoints, "h0-matched", "runs")
    with pytest.raises(SystemExit, match="needs its own --runs-dir"):
        run.main(argv)
