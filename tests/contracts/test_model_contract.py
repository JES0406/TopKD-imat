"""Contract tests for the Model interface (docs/interfaces.md #1).

Written by: Experiments subteam (consumer). Satisfied by: Models subteam (provider).
TinyNet (tests/contracts/tinynet.py) is the stub model Experiments builds against.
"""

import pytest
import torch
from torch import nn

from tests.contracts.tinynet import TinyNet
from topkd_imat import models
from topkd_imat.training.contracts import ModelContractError, check_model, check_model_pair


def test_model_returns_logits_and_one_cloud_per_layer_pair():
    x = torch.randn(4, 3, 32, 32)
    n_pairs = check_model_pair(TinyNet(width=16), TinyNet(width=4), x, num_classes=10)
    assert n_pairs == 2


def test_models_package_declares_checkpoints_and_normalization():
    for name in ("TEACHER_CHECKPOINT", "STUDENT_CHECKPOINT"):
        value = getattr(models, name)
        assert value is None or isinstance(value, str), f"{name} must be a str or None"
    mean, std = models.INPUT_NORMALIZATION
    assert len(mean) == len(std) == 3
    assert all(s > 0 for s in std)


class _ReturnsOnlyLogits(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 10)


class _UnflattenedFeatures(TinyNet):
    def forward(self, x):
        logits, _ = super().forward(x)
        return logits, [self.block1(x)]


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_ReturnsOnlyLogits(), "must return a pair"),
        (_UnflattenedFeatures(), r"features\[0\] must have shape \(4, dim\)"),
    ],
)
def test_off_contract_models_fail_with_a_readable_message(model, message):
    with pytest.raises(ModelContractError, match=message):
        check_model(model, torch.randn(4, 3, 32, 32), num_classes=10)


def test_teacher_and_student_must_have_the_same_number_of_layer_pairs():
    class OnePair(TinyNet):
        def forward(self, x):
            logits, feats = super().forward(x)
            return logits, feats[-1:]

    with pytest.raises(ModelContractError, match="lengths must match"):
        check_model_pair(TinyNet(), OnePair(), torch.randn(4, 3, 32, 32), num_classes=10)
