import torch

from topkd_imat.training.probe import fit_probe


def test_probe_separates_linearly_separable_features():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(3, 16, generator=g) * 5
    labels = torch.arange(3).repeat(100)
    feats = centers[labels] + torch.randn(300, 16, generator=g)
    probe = fit_probe(feats, labels, num_classes=3)
    with torch.no_grad():
        acc = (probe(feats).argmax(1) == labels).float().mean()
    assert acc > 0.95
