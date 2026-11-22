import torch

from topkd_imat.training.kd import distill_kl


def test_kd_is_zero_when_student_matches_teacher():
    logits = torch.randn(8, 10)
    assert distill_kl(logits.clone(), logits, T=4.0).abs() < 1e-6


def test_kd_is_positive_and_trains_only_the_student():
    teacher = torch.randn(8, 10, requires_grad=True)
    student = torch.randn(8, 10, requires_grad=True)
    loss = distill_kl(student, teacher, T=4.0)
    loss.backward()
    assert loss > 0
    assert student.grad is not None and teacher.grad is None


def test_kd_matches_reference_formula():
    s, t, T = torch.randn(4, 10), torch.randn(4, 10), 2.0
    p_t = torch.softmax(t / T, 1)
    expected = (p_t * (p_t.log() - torch.log_softmax(s / T, 1))).sum(1).mean() * T**2
    assert torch.allclose(distill_kl(s, t, T), expected, atol=1e-6)
