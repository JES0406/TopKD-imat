"""Teacher and Student networks.

Owner: Models subteam (W1, W2).
Provides the Model interface (docs/interfaces.md #1): ``logits, features = model(x)``.

Teacher: ResNet-56 (CIFAR-10). Student: ResNet-8 pruned from a CIFAR-100 ResNet-20.
Three Layer Pairs, one per stage (dims 16, 32, 64 in both networks). See
docs/adr/0003-resnet56-teacher-pruned-resnet8-student.md and ``build.py``.

The constants below tell Experiments which models to train with. Each checkpoint is a whole
model saved with ``torch.save(model, path)`` (not a state_dict); its class must live in this
package and keep its name, so ``torch.load(path, weights_only=False)`` can rebuild it.
Build both with ``uv run python scripts/build_models.py`` (writes checkpoints/).
"""

from topkd_imat.models.resnet_cifar import CifarResNet, cifar_resnet, prune_depth, replace_head

__all__ = [
    "INPUT_NORMALIZATION",
    "STUDENT_CHECKPOINT",
    "TEACHER_CHECKPOINT",
    "CifarResNet",
    "cifar_resnet",
    "prune_depth",
    "replace_head",
]

# Where the trained Teacher / pretrained Student live: a local path or "hf://<user>/<repo>/<file>"
# (downloaded from the Hugging Face Hub on first use). Kept None so `make test` does not need
# the files on every machine. Build them (deterministically) with scripts/build_models.py, then
# --teacher-checkpoint checkpoints/teacher_resnet56.pt
# --student-checkpoint checkpoints/student_resnet8_from_c100_resnet20.pt
TEACHER_CHECKPOINT: str | None = None
STUDENT_CHECKPOINT: str | None = None

# Per-channel (mean, std) the models expect their inputs normalised with: the values the
# Teacher (chenyaofo cifar10_resnet56) was trained with, taken from its training log. The std
# is not the true CIFAR-10 pixel std (0.2470, 0.2435, 0.2616); with that one the Teacher drops
# from 94.37 % to 94.25 % test accuracy (scripts/evaluate_models.py). The Student comes from a
# CIFAR-100 model (other stats) but is trained on CIFAR-10, so its BatchNorms adapt to these.
INPUT_NORMALIZATION: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010),
)
