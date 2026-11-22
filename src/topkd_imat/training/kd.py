"""Knowledge Distillation loss (Hinton et al. 2015), as in the TopKD reference code."""

import torch
import torch.nn.functional as F


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 4.0):
    """KL(softmax(teacher/T) || softmax(student/T)) * T^2, averaged over the batch.

    The T^2 factor keeps the gradient scale comparable to cross-entropy when T changes.
    """
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits.detach() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="sum") * (T**2) / student_logits.shape[0]
