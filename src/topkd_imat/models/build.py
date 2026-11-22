"""Build the Teacher and the pretrained Student from chenyaofo/pytorch-cifar-models weights.

Decisions and their reasons: docs/adr/0003-resnet56-teacher-pruned-resnet8-student.md.

- Teacher: ResNet-56 trained on CIFAR-10 (94.37 % test accuracy reported upstream).
- Student: ResNet-20 trained on CIFAR-100 (68.83 % on CIFAR-100 upstream), depth-pruned to a
  ResNet-8 (first block of each stage kept) and given a fresh 10-class head. It has never
  seen CIFAR-10 labels. The ``scratch`` Student is this same ResNet-8 with its weights
  reset (``topkd_imat.training.checkpoints.reset_weights``).

Weights are downloaded once into the torch hub cache; the hash in each file name is checked.
"""

import torch

from topkd_imat.models.resnet_cifar import CifarResNet, cifar_resnet, prune_depth, replace_head

_RELEASES = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet"
WEIGHT_URLS = {
    ("cifar10", 20): f"{_RELEASES}/cifar10_resnet20-4118986f.pt",
    ("cifar10", 56): f"{_RELEASES}/cifar10_resnet56-187c023a.pt",
    ("cifar100", 20): f"{_RELEASES}/cifar100_resnet20-23dac2f1.pt",
    ("cifar100", 56): f"{_RELEASES}/cifar100_resnet56-f2eff4c8.pt",
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

# Per-channel (mean, std) each upstream model was trained with (from chenyaofo's training logs).
UPSTREAM_NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
}

TEACHER_DEPTH = 56
STUDENT_SOURCE_DEPTH = 20
STUDENT_BLOCKS_PER_STAGE = 1  # -> ResNet-8


def load_upstream(dataset: str, depth: int, progress: bool = True) -> CifarResNet:
    """A chenyaofo CIFAR ResNet with its released weights (``dataset``: cifar10 / cifar100)."""
    model = cifar_resnet(depth, NUM_CLASSES[dataset])
    state = torch.hub.load_state_dict_from_url(
        WEIGHT_URLS[(dataset, depth)], map_location="cpu", progress=progress, check_hash=True
    )
    model.load_state_dict(state, strict=True)
    return model


def build_teacher(progress: bool = True) -> CifarResNet:
    """ResNet-56 trained on CIFAR-10, in eval mode."""
    return load_upstream("cifar10", TEACHER_DEPTH, progress).eval()


def build_student(progress: bool = True) -> CifarResNet:
    """ResNet-8 pruned from the CIFAR-100 ResNet-20, with a fresh 10-class head."""
    model = load_upstream("cifar100", STUDENT_SOURCE_DEPTH, progress)
    prune_depth(model, STUDENT_BLOCKS_PER_STAGE)
    return replace_head(model, num_classes=10)
