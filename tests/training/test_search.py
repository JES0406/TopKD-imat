"""The w_kd x w_top search: plan, per-configuration folders, resume, validation-based summary."""

import csv
import json

import pytest
import torch

from tests.contracts.tinynet import TinyNet
from topkd_imat.training.search import W_KDS, W_TOPS, main, plan, run_argv


def test_plan_crosses_w_kd_with_w_top_and_gives_each_w_kd_a_kd_control():
    runs = plan(W_KDS, W_TOPS, seeds=(0, 1))
    assert len(runs) == 2 * len(W_KDS) * (1 + len(W_TOPS)) == 24
    configs = {r["config"] for r in runs}
    assert {"KD_wkd1", "KD_wkd2", "KD_wkd4"} <= configs
    assert {"h0-pi-norm_wkd1_wtop3", "h0-pi-norm_wkd4_wtop1"} <= configs
    assert not any("matched" in c for c in configs)  # only h0-pi-norm for now
    assert len({(r["config"], r["seed"]) for r in runs}) == len(runs)


def test_each_configuration_gets_its_own_runs_dir(tmp_path):
    kd, top = plan((2.0,), (10.0,), seeds=(3,))
    argv = run_argv(top, tmp_path, ["--epochs", "1"])
    value = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
    assert value("--runs-dir") == str(tmp_path / "h0-pi-norm_wkd2_wtop10")
    assert (value("--w-kd"), value("--w-top"), value("--top-loss")) == ("2", "10", "h0-pi-norm")
    assert (value("--condition"), value("--init"), value("--seed")) == ("KD+Top", "scratch", "3")
    assert argv[-2:] == ["--epochs", "1"]
    control = run_argv(kd, tmp_path, [])
    assert control[control.index("--condition") + 1] == "KD"
    assert control[control.index("--w-kd") + 1] == "2"
    assert "--top-loss" not in control and "--w-top" not in control


@pytest.mark.parametrize("flag", ["--w-top", "--w-kd", "--top-loss", "--seed"])
def test_flags_the_search_sets_are_refused(tmp_path, flag):
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--dry-run", flag, "5"])


@pytest.fixture
def checkpoints(tmp_path):
    torch.manual_seed(0)
    torch.save(TinyNet(width=8), tmp_path / "teacher.pt")
    torch.save(TinyNet(width=4), tmp_path / "student.pt")
    return tmp_path


def test_search_trains_resumes_and_ranks_by_validation(checkpoints, capsys):
    root = checkpoints / "search"
    argv = [
        "--root", str(root), "--w-kds", "2", "--w-tops", "1", "10",
        "--dataset", "fake", "--fake-size", "256", "--num-workers", "0", "--device", "cpu",
        "--epochs", "1", "--pi-calibration-batches", "2",
        "--teacher-checkpoint", str(checkpoints / "teacher.pt"),
        "--student-checkpoint", str(checkpoints / "student.pt"),
    ]  # fmt: skip
    assert main(argv) == 0
    expected = {
        "KD_wkd2": ("KD", None),
        "h0-pi-norm_wkd2_wtop1": ("KD+Top", 1.0),
        "h0-pi-norm_wkd2_wtop10": ("KD+Top", 10.0),
    }
    for config, (condition, w_top) in expected.items():
        (cfg_path,) = (root / config).glob("*/config.json")
        cfg = json.loads(cfg_path.read_text())
        assert (cfg["condition"], cfg["init"], cfg["w_kd"]) == (condition, "scratch", 2.0)
        if w_top is not None:
            assert (cfg["w_top"], cfg["top_loss"]) == (w_top, "h0-pi-norm")

    with (root / "summary.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["config"] for r in rows} == set(expected)
    accs = [float(r["val_acc_last"]) for r in rows]
    assert accs == sorted(accs, reverse=True)
    assert "accuracy" not in rows[0]  # the test accuracy is never used to rank

    capsys.readouterr()
    assert main(argv) == 0  # resumes: every run is already in its results.csv
    assert capsys.readouterr().out.count("skipping") == 3
