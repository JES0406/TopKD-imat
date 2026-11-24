import csv

import torch
from torch import nn

from tests.contracts.tinynet import TinyNet
from topkd_imat.training import probe_reference
from topkd_imat.training.probe_reference import recompute_batchnorm


class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(4, 10)

    def forward(self, x):
        f = self.bn(self.conv(x)).mean(dim=(2, 3))
        return self.head(f), [f]


def test_recompute_batchnorm_averages_over_the_loader_and_keeps_momentum():
    torch.manual_seed(0)
    model = BNNet()
    x = torch.randn(32, 3, 8, 8) * 3 + 5
    loader = [(x[:16], None), (x[16:], None)]
    assert recompute_batchnorm(model, loader, "cpu") == 1
    expected = model.conv(x).mean(dim=(0, 2, 3))
    assert torch.allclose(model.bn.running_mean, expected, atol=1e-4)
    assert model.bn.momentum == 0.1 and not model.training


def test_recompute_batchnorm_without_batchnorm_does_nothing():
    model = TinyNet()
    before = [p.clone() for p in model.state_dict().values()]
    assert recompute_batchnorm(model, [(torch.randn(4, 3, 32, 32), None)], "cpu") == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.state_dict().values(), strict=True))


def test_main_probes_teacher_untrained_students_and_saved_students(tmp_path):
    torch.manual_seed(0)
    torch.save(TinyNet(width=8), tmp_path / "teacher.pt")
    torch.save(TinyNet(width=4), tmp_path / "student.pt")
    runs = tmp_path / "runs"
    (runs / "CIFAR-10_Top_label-free_scratch_s0").mkdir(parents=True)
    torch.save(TinyNet(width=4), runs / "CIFAR-10_Top_label-free_scratch_s0" / "student.pt")

    rows = probe_reference.main(
        [
            "--dataset", "fake", "--fake-size", "64", "--num-workers", "0",
            "--device", "cpu", "--runs-dir", str(runs), "--trained", "--seeds", "0", "1",
            "--teacher-checkpoint", str(tmp_path / "teacher.pt"),
            "--student-checkpoint", str(tmp_path / "student.pt"),
        ]
    )  # fmt: skip

    names = [r["model"] for r in rows]
    assert names == [
        "teacher",
        "student_pretrained_untrained",
        "student_scratch_untrained_s0",
        "student_scratch_untrained_s1",
        "CIFAR-10_Top_label-free_scratch_s0",
    ]  # TinyNet has no BatchNorm, so no "recomputed" rows
    assert all(0.0 <= r["probe_accuracy"] <= 1.0 for r in rows)
    with (runs / "probe_reference.csv").open() as f:
        assert [r["model"] for r in csv.DictReader(f)] == names
