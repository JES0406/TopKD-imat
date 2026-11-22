"""Checkpoint loading. The real-model tests run only on a developer machine where the
Models subteam has declared its checkpoints and CIFAR-10 is already downloaded."""

import os
from pathlib import Path

import pytest
import torch

from tests.contracts.tinynet import TinyNet
from topkd_imat import models
from topkd_imat.training.checkpoints import (
    CheckpointError,
    load_model,
    reset_weights,
    resolve_checkpoint,
)
from topkd_imat.training.contracts import check_model_pair


def test_whole_model_round_trips(tmp_path):
    model = TinyNet()
    torch.save(model, tmp_path / "m.pt")
    loaded = load_model(str(tmp_path / "m.pt"), "teacher")
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(model(x)[0], loaded(x)[0])


def test_state_dict_is_rejected_with_a_hint(tmp_path):
    torch.save(TinyNet().state_dict(), tmp_path / "sd.pt")
    with pytest.raises(CheckpointError, match="not its state_dict"):
        load_model(str(tmp_path / "sd.pt"), "student")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (None, "No teacher checkpoint is set"),
        ("does/not/exist.pt", "does not exist"),
        ("hf://only-user", "must look like"),
    ],
)
def test_bad_specs_fail_with_a_readable_message(spec, message):
    with pytest.raises(CheckpointError, match=message):
        resolve_checkpoint(spec, "teacher")


def test_scratch_reset_changes_every_layer():
    model = TinyNet()
    before = [p.detach().clone() for p in model.parameters()]
    assert reset_weights(model) == 3
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


real = pytest.mark.skipif(
    models.TEACHER_CHECKPOINT is None
    or models.STUDENT_CHECKPOINT is None
    or os.environ.get("CI") == "true",
    reason="needs TEACHER_CHECKPOINT/STUDENT_CHECKPOINT in topkd_imat.models; skipped in CI",
)


@real
def test_declared_models_load_and_follow_the_contract():
    teacher = load_model(models.TEACHER_CHECKPOINT, "teacher").eval()
    student = load_model(models.STUDENT_CHECKPOINT, "student").eval()
    with torch.no_grad():
        assert check_model_pair(teacher, student, torch.randn(4, 3, 32, 32), 10) >= 1


@real
def test_declared_teacher_is_accurate_on_cifar10():
    if not (Path("data") / "cifar-10-batches-py").is_dir():
        pytest.skip("CIFAR-10 not in data/; any training run downloads it")
    from topkd_imat.training.data import make_loaders

    loaders = make_loaders("CIFAR-10", models.INPUT_NORMALIZATION, num_workers=0)
    teacher = load_model(models.TEACHER_CHECKPOINT, "teacher").eval()
    correct = seen = 0
    with torch.no_grad():
        for x, y in loaders.test:
            correct += int((teacher(x)[0].argmax(1) == y).sum())
            seen += len(y)
            if seen >= 512:
                break
    assert correct / seen > 0.8, f"Teacher accuracy {correct / seen:.3f} on {seen} test images"
