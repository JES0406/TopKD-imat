import torch

from topkd_imat.training.data import make_loaders

NORM = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))


def test_fake_test_images_never_appear_in_the_training_split():
    loaders = make_loaders("fake", NORM, fake_size=64, num_workers=0)
    train = torch.cat([x for x, _ in loaders.train_eval] + [x for x, _ in loaders.val])
    test = torch.cat([x for x, _ in loaders.test])
    assert not any(torch.equal(t, s) for t in test for s in train)
