"""Checkpoint loading and model building. The real-model tests are slow (they download the
weights on first use) and run with `make test-e2e`, never in CI."""

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
from topkd_imat.training.run import get_model


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


def test_without_a_checkpoint_the_models_are_built(monkeypatch):
    built = {}

    def fake_builder(role):
        def build(progress=True):
            built[role] = True
            return TinyNet()

        return build

    monkeypatch.setattr(models, "build_teacher", fake_builder("teacher"))
    monkeypatch.setattr(models, "build_student", fake_builder("student"))
    teacher, t_source = get_model(None, "teacher")
    student, s_source = get_model(None, "student")
    assert built == {"teacher": True, "student": True}
    assert t_source == {"built": "build_teacher()"}
    assert s_source["built"] == "build_student()" and "revision" in s_source["healed_weights"]


def test_a_given_checkpoint_is_loaded_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "build_teacher", lambda **_: pytest.fail("must not build"))
    torch.save(TinyNet(), tmp_path / "t.pt")
    model, source = get_model(str(tmp_path / "t.pt"), "teacher")
    assert isinstance(model, TinyNet) and source == {"checkpoint": str(tmp_path / "t.pt")}


# The real Teacher and Student, as training gets them (declared checkpoint, else built and
# downloaded on first use). Slow and needs the network: run with `make test-e2e`, never in CI.
def _teacher_and_student():
    teacher, _ = get_model(models.TEACHER_CHECKPOINT, "teacher")
    student, _ = get_model(models.STUDENT_CHECKPOINT, "student")
    return teacher.eval(), student.eval()


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="downloads weights; not in CI")
def test_declared_models_load_and_follow_the_contract():
    teacher, student = _teacher_and_student()
    with torch.no_grad():
        assert check_model_pair(teacher, student, torch.randn(4, 3, 32, 32), 10) >= 1


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="downloads weights; not in CI")
def test_declared_teacher_is_accurate_on_cifar10():
    if not (Path("data") / "cifar-10-batches-py").is_dir():
        pytest.skip("CIFAR-10 not in data/; any training run downloads it")
    from topkd_imat.training.data import make_loaders

    loaders = make_loaders("CIFAR-10", models.INPUT_NORMALIZATION, num_workers=0)
    teacher, _ = _teacher_and_student()
    correct = seen = 0
    with torch.no_grad():
        for x, y in loaders.test:
            correct += int((teacher(x)[0].argmax(1) == y).sum())
            seen += len(y)
            if seen >= 512:
                break
    assert correct / seen > 0.8, f"Teacher accuracy {correct / seen:.3f} on {seen} test images"
