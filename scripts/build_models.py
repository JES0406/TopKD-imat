"""Build the Teacher and pretrained Student checkpoints (W1, W2) and check them.

    uv run python scripts/build_models.py                      # writes checkpoints/*.pt
    uv run python scripts/build_models.py --upload <user>/<repo>   # also pushes them to the HF Hub

Each checkpoint is a whole model (``torch.save(model)``), as topkd_imat.training expects.
After uploading, set TEACHER_CHECKPOINT / STUDENT_CHECKPOINT in topkd_imat/models/__init__.py
to the printed ``hf://`` specs.
"""

import argparse
from pathlib import Path

import torch

from topkd_imat.models.build import build_student, build_teacher
from topkd_imat.training.checkpoints import load_model
from topkd_imat.training.contracts import check_model_pair

FILES = {
    "teacher": "teacher_resnet56.pt",
    "student": "student_resnet8_from_c100_resnet20.pt",
}


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--upload", metavar="USER/REPO", help="Hugging Face model repo to upload to")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = {"teacher": build_teacher(), "student": build_student()}
    for role, model in models.items():
        path = out / FILES[role]
        torch.save(model, path)
        print(
            f"{role}: depth {model.depth}, blocks/stage {model.blocks_per_stage}, "
            f"{n_params(model):,} params -> {path}"
        )

    # Reload exactly as the training code does, and check the Model interface on the pair.
    teacher = load_model(str(out / FILES["teacher"]), "teacher").eval()
    student = load_model(str(out / FILES["student"]), "student").eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        n_pairs = check_model_pair(teacher, student, x, num_classes=10)
        dims_t = [f.shape[1] for f in teacher(x)[1]]
        dims_s = [f.shape[1] for f in student(x)[1]]
    print(f"Model interface OK: {n_pairs} Layer Pairs, Teacher dims {dims_t}, Student {dims_s}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload, repo_type="model", exist_ok=True)
        for role, name in FILES.items():
            api.upload_file(path_or_fileobj=out / name, path_in_repo=name, repo_id=args.upload)
            print(f'{role.upper()}_CHECKPOINT = "hf://{args.upload}/{name}"')


if __name__ == "__main__":
    main()
