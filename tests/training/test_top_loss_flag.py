from topkd_imat.topology import LOSSES
from topkd_imat.training.run import build_parser

BASE = ["--condition", "Top", "--supervision", "label-free", "--init", "scratch"]


def test_top_loss_defaults_to_the_persistence_image_loss():
    assert build_parser().parse_args(BASE).top_loss == "h0-pi"


def test_top_loss_accepts_every_registered_loss():
    for name in LOSSES:
        assert build_parser().parse_args([*BASE, "--top-loss", name]).top_loss == name
