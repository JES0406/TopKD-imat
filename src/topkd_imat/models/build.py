"""Build the Teacher and the pretrained Student.

Decisions and their reasons: docs/adr/0003-resnet56-teacher-pruned-resnet8-student.md.

- Teacher: chenyaofo's ResNet-56 trained on CIFAR-10 (94.37 % test accuracy).
- Student: chenyaofo's ResNet-20 trained on CIFAR-100, depth-pruned to a ResNet-8 (first
  block of each stage kept), retrained ("healed") by us on CIFAR-100 with
  scripts/heal_student_cifar100.py, and given a fresh 10-class head. It has never seen
  CIFAR-10 labels. The healed weights live on the Hugging Face Hub, pinned to one revision.
  The ``scratch`` Student is this same ResNet-8 with its weights reset
  (``topkd_imat.training.checkpoints.reset_weights``).

Weights are downloaded once and cached (torch hub for chenyaofo's, with the hash in each file
name checked; the HF cache for the healed Student). Both builds are deterministic: the files
are fixed and the Student's new head is seeded, so every machine gets identical models.
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
STUDENT_DEPTH = 8
STUDENT_HEAD_SEED = 0  # the fresh 10-class head is the only random part of either model

# Healed ResNet-8 (100-class head) made by scripts/heal_student_cifar100.py, seed 0, on an
# RTX 2060: 57.45 % on CIFAR-100 test. Pinned to a revision so a later upload cannot change it.
HEALED_STUDENT = {
    "repo_id": "MarioAlonso7/topkd",
    "filename": "student_resnet8_healed_c100_state_dict.pt",
    "revision": "d4a68ebce3c2fa1b8c089e800f498d0c56972f11",
    "sha256": "e26d8e6d7e9850e5853035a4a76b5db4ec38a94a47c035e82291ba2b82e6dd80",
}


def load_upstream(dataset: str, depth: int, progress: bool = True) -> CifarResNet:
    """A chenyaofo CIFAR ResNet with its released weights (``dataset``: cifar10 / cifar100)."""
    model = cifar_resnet(depth, NUM_CLASSES[dataset])
    state = torch.hub.load_state_dict_from_url(
        WEIGHT_URLS[(dataset, depth)], map_location="cpu", progress=progress, check_hash=True
    )
    model.load_state_dict(state, strict=True)
    return model


def healed_weights_path() -> str:
    """Local path of the healed Student's state_dict, downloaded from the Hub on first use."""
    import hashlib

    from huggingface_hub import hf_hub_download

    spec = HEALED_STUDENT
    path = hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"])
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"{path} has sha256 {digest}, expected {spec['sha256']}.")
    return path


def build_teacher(progress: bool = True) -> CifarResNet:
    """ResNet-56 trained on CIFAR-10, in eval mode."""
    return load_upstream("cifar10", TEACHER_DEPTH, progress).eval()


def build_student(
    progress: bool = True, seed: int = STUDENT_HEAD_SEED, healed: bool = True, healed_weights=None
) -> CifarResNet:
    """The pretrained Student: a ResNet-8 with a fresh, seeded 10-class head.

    ``healed=True`` (default, the team's choice): the pruned ResNet-8 retrained on CIFAR-100,
    from the Hub (or from a local ``healed_weights`` state_dict). ``healed=False``: the pruned
    ResNet-8 straight from chenyaofo's CIFAR-100 ResNet-20, not retrained (kept for comparison).
    """
    if healed:
        model = cifar_resnet(STUDENT_DEPTH, NUM_CLASSES["cifar100"])
        path = healed_weights or healed_weights_path()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    else:
        model = load_upstream("cifar100", STUDENT_SOURCE_DEPTH, progress)
        prune_depth(model, STUDENT_BLOCKS_PER_STAGE)
    return replace_head(model, num_classes=10, seed=seed)
