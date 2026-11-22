"""Teacher / Student architecture (W1, W2). No downloads: random weights only."""

import re

import pytest
import torch

from topkd_imat.models.resnet_cifar import (
    STAGE_WIDTHS,
    cifar_resnet,
    prune_depth,
    replace_head,
)
from topkd_imat.training.checkpoints import load_model, reset_weights
from topkd_imat.training.contracts import check_model_pair


def test_package_exports_the_builders_training_calls():
    from topkd_imat import models

    assert callable(models.build_teacher) and callable(models.build_student)


@pytest.fixture
def x():
    return torch.randn(4, 3, 32, 32)


def test_teacher_and_student_shapes_follow_the_model_interface(x):
    teacher, student = cifar_resnet(56).eval(), cifar_resnet(8).eval()
    with torch.no_grad():
        assert check_model_pair(teacher, student, x, num_classes=10) == 3
        for model in (teacher, student):
            _, feats = model(x)
            assert [f.shape[1] for f in feats] == list(STAGE_WIDTHS)


@pytest.mark.parametrize("depth", [8, 20, 56])
def test_last_cloud_is_exactly_the_classifier_input(x, depth):
    model = cifar_resnet(depth).eval()
    with torch.no_grad():
        logits, feats = model(x)
        assert torch.allclose(model.fc(feats[-1]), logits)


@pytest.mark.parametrize(("depth", "blocks"), [(8, 1), (20, 3), (56, 9)])
def test_depth_counts_weighted_layers(depth, blocks):
    model = cifar_resnet(depth)
    assert model.blocks_per_stage == (blocks,) * 3
    assert model.depth == depth


def test_bad_depth_is_rejected():
    with pytest.raises(ValueError, match="6n\\+2"):
        cifar_resnet(10)


def test_pruning_resnet20_gives_a_resnet8_with_the_first_blocks_weights():
    r20 = cifar_resnet(20, num_classes=100)
    first_blocks = {
        k: v.clone()
        for k, v in r20.state_dict().items()
        if not re.match(r"layer\d\.[12]\.", k)  # drop blocks 1 and 2 of every stage
    }
    pruned = prune_depth(r20, blocks_per_stage=1)
    reference = cifar_resnet(8, num_classes=100).state_dict()
    state = pruned.state_dict()
    assert state.keys() == reference.keys()
    assert all(state[k].shape == reference[k].shape for k in state)
    assert all(torch.equal(state[k], first_blocks[k]) for k in state)
    assert pruned.depth == 8


def test_pruning_cannot_remove_a_stage_or_add_blocks():
    with pytest.raises(ValueError):
        prune_depth(cifar_resnet(20), 0)
    with pytest.raises(ValueError):
        prune_depth(cifar_resnet(8), 2)


def test_replace_head_changes_only_the_classifier(x):
    model = cifar_resnet(8, num_classes=100).eval()  # eval: no BatchNorm stat updates
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("fc.")}
    replace_head(model, 10)
    assert model(x)[0].shape == (4, 10)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_seeded_head_is_reproducible_and_leaves_global_rng_alone():
    a = replace_head(cifar_resnet(8, num_classes=100), 10, seed=0).fc
    b = replace_head(cifar_resnet(8, num_classes=100), 10, seed=0).fc
    c = replace_head(cifar_resnet(8, num_classes=100), 10, seed=1).fc
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
    assert not torch.equal(a.weight, c.weight)

    model = cifar_resnet(8, num_classes=100)
    state = torch.random.get_rng_state()
    replace_head(model, 10, seed=0)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_student_checkpoint_round_trips_and_resets_for_scratch(tmp_path, x):
    student = replace_head(prune_depth(cifar_resnet(20, num_classes=100), 1), 10).eval()
    with torch.no_grad():  # stand-in for pretrained weights (fresh BatchNorm is already 1 / 0)
        for p in student.parameters():
            p.add_(torch.randn_like(p))
    torch.save(student, tmp_path / "student.pt")
    loaded = load_model(str(tmp_path / "student.pt"), "student").eval()
    with torch.no_grad():
        assert torch.equal(student(x)[0], loaded(x)[0])
    before = [p.detach().clone() for p in loaded.parameters()]
    reset_weights(loaded)
    changed = [not torch.equal(a, b) for a, b in zip(before, loaded.parameters(), strict=True)]
    assert all(changed)  # convs / linears redrawn, BatchNorm back to 1 / 0: nothing pretrained left
