"""Build the Teacher and pretrained Student checkpoints (W1, W2) and check them.

    uv run python scripts/build_models.py        # writes checkpoints/*.pt

Run it once on every machine that trains (e.g. the DGX). The build is deterministic: the
Teacher is chenyaofo's released ResNet-56 and the Student's only random part (its new head)
is seeded, so every machine gets the same models. Each checkpoint is a whole model
(``torch.save(model)``), as topkd_imat.training expects; pass the printed paths with
--teacher-checkpoint / --student-checkpoint.
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
    print(
        f"Train with: --teacher-checkpoint {out / FILES['teacher']} "
        f"--student-checkpoint {out / FILES['student']}"
    )


if __name__ == "__main__":
    main()
