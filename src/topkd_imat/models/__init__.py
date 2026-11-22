"""Teacher and Student networks.

Owner: Models subteam (W1, W2).
Provides the Model interface (docs/interfaces.md #1): ``logits, features = model(x)``.

The constants below tell Experiments which models to train with. Each checkpoint is a whole
model saved with ``torch.save(model, path)`` (not a state_dict); its class must live in this
package and keep its name, so ``torch.load(path, weights_only=False)`` can rebuild it.
"""

# Where the trained Teacher / pretrained Student live: a local path or "hf://<user>/<repo>/<file>"
# (downloaded from the Hugging Face Hub on first use). None = not available yet.
TEACHER_CHECKPOINT: str | None = None
STUDENT_CHECKPOINT: str | None = None

# Per-channel (mean, std) the models expect their inputs normalised with. Standard CIFAR-10
# training-set statistics; change them if the checkpoints were trained with other values.
INPUT_NORMALIZATION: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616),
)
