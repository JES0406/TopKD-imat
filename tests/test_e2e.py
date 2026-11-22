"""End-to-end: the whole grid on TinyNet and fake data. Slow-ish; not part of `make test`.

make test-e2e
"""

import copy
import json
import warnings

import numpy as np
import pytest
import torch

from tests.contracts.tinynet import TinyNet
from topkd_imat.topology import RandomTopLoss
from topkd_imat.training import grid
from topkd_imat.training.data import make_loaders
from topkd_imat.training.loop import RunConfig, train_run
from topkd_imat.training.results import read_rows, run_name

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.filterwarnings("ignore:RandomTopLoss is a placeholder"),
]
NORM = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))


@pytest.fixture
def checkpoints(tmp_path):
    torch.manual_seed(0)
    torch.save(TinyNet(width=16), tmp_path / "teacher.pt")
    torch.save(TinyNet(width=4), tmp_path / "student.pt")
    return tmp_path


def test_full_grid_on_fake_data(checkpoints):
    runs_dir = checkpoints / "runs"
    common = [
        "--dataset", "fake",
        "--runs-dir", str(runs_dir),
        "--epochs", "1",
        "--num-workers", "0",
        "--device", "cpu",
        "--teacher-checkpoint", str(checkpoints / "teacher.pt"),
        "--student-checkpoint", str(checkpoints / "student.pt"),
    ]  # fmt: skip
    assert grid.main(["--inits", "pretrained", "scratch", *common]) == 0

    rows = read_rows(runs_dir / "results.csv")
    assert len(rows) == 14
    assert (runs_dir / "teacher" / "clouds_test.npz").is_file()
    for r in rows:
        d = runs_dir / run_name(r)
        for name in ("config.json", "metrics.csv", "student.pt", "clouds_test.npz"):
            assert (d / name).is_file(), f"{d.name} is missing {name}"
        assert any((d / "tb").iterdir())
        with np.load(d / "clouds_test.npz") as clouds:
            assert {"labels", "pair_0", "pair_1"} <= set(clouds.files)
            assert clouds["pair_1"].dtype == np.float16
        config = json.loads((d / "config.json").read_text())
        assert config["condition"] == r["condition"] and "git_commit" in config

    # Re-running the finished grid trains nothing.
    assert grid.main(["--inits", "pretrained", "scratch", *common]) == 0
    assert len(read_rows(runs_dir / "results.csv")) == 14


@pytest.mark.parametrize(
    ("condition", "supervision"),
    [("Baseline", "label-supervised"), ("KD", "label-free"), ("Top", "label-free")],
)
def test_student_learns_and_teacher_stays_frozen(tmp_path, condition, supervision):
    torch.manual_seed(0)
    teacher, student = TinyNet(width=16), TinyNet(width=4)
    t_before, s_before = copy.deepcopy(teacher.state_dict()), copy.deepcopy(student.state_dict())
    loaders = make_loaders("fake", NORM, batch_size=16, num_workers=0, fake_size=64)
    cfg = RunConfig(
        dataset="fake",
        condition=condition,
        supervision=supervision,
        init="pretrained",
        epochs=1,
        runs_dir=str(tmp_path),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_run(cfg, teacher, student, loaders, RandomTopLoss(seed=0))
    assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in t_before.items())
    assert any(not torch.equal(v, student.state_dict()[k]) for k, v in s_before.items())
