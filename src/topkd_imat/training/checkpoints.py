"""Load the Teacher / Student declared in ``topkd_imat.models``.

A checkpoint spec is a local path or ``hf://<user>/<repo>/<file>``. Hub files are looked up
in the local Hugging Face cache first and downloaded only if missing.
"""

from pathlib import Path

import torch
from torch import nn

HF_PREFIX = "hf://"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unreachable or not a whole model."""


def resolve_checkpoint(spec: str | None, role: str) -> Path:
    """Return a local file for ``spec``, downloading it from the Hub if needed."""
    if spec is None:
        raise CheckpointError(
            f"No {role} checkpoint is set. The Models subteam declares it in "
            f"topkd_imat/models/__init__.py ({role.upper()}_CHECKPOINT), or pass "
            f"--{role}-checkpoint on the command line."
        )
    if not spec.startswith(HF_PREFIX):
        path = Path(spec)
        if not path.is_file():
            raise CheckpointError(f"{role} checkpoint {spec!r} does not exist on this machine.")
        return path
    parts = spec[len(HF_PREFIX) :].split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise CheckpointError(
            f"{role} checkpoint {spec!r} must look like hf://<user>/<repo>/<file>."
        )
    repo_id, filename = f"{parts[0]}/{parts[1]}", parts[2]
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(hf_hub_download(repo_id, filename, local_files_only=True))
    except LocalEntryNotFoundError:
        pass
    try:
        return Path(hf_hub_download(repo_id, filename))
    except Exception as err:  # network, auth, missing file: all need the same human fix
        raise CheckpointError(
            f"Could not download the {role} checkpoint {spec!r}: {err}. Check the repo name, "
            f"your network and, for a private repo, `huggingface-cli login`."
        ) from err


def load_model(spec: str | None, role: str, device: str | torch.device = "cpu") -> nn.Module:
    """Load a whole model saved with ``torch.save(model, path)``.

    ``weights_only=False`` unpickles arbitrary objects: only load checkpoints from the team.
    """
    path = resolve_checkpoint(spec, role)
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except (AttributeError, ModuleNotFoundError) as err:
        raise CheckpointError(
            f"{role} checkpoint {path} refers to a class that cannot be imported ({err}). The "
            f"model class must stay in topkd_imat.models under the same name it was saved with."
        ) from err
    if not isinstance(model, nn.Module):
        raise CheckpointError(
            f"{role} checkpoint {path} holds a {type(model).__name__}, not a model. Save the "
            f"whole model with torch.save(model, path), not its state_dict."
        )
    return model


def reset_weights(model: nn.Module) -> int:
    """Re-initialise every layer in place (the ``scratch`` Initialisation); return how many."""
    n = 0
    for module in model.modules():
        if module is not model and hasattr(module, "reset_parameters"):
            module.reset_parameters()
            n += 1
    if n == 0:
        raise CheckpointError("reset_weights found no layer with reset_parameters().")
    return n
