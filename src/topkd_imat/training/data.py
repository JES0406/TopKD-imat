"""Data loaders for training: the project's definitive loaders (agreed 2026-11-22; there is
no separate ``topkd_imat.data``).

Train split: random crop + horizontal flip. Validation: a fixed 5k slice of the training
set (same for every run), with test-time transforms. ``fake`` is torchvision's FakeData,
for tests that must not download anything.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

VAL_SPLIT_SEED = 0
NUM_CLASSES = {"CIFAR-10": 10, "fake": 10}


@dataclass
class Loaders:
    train: DataLoader  # augmented, shuffled, drop_last (a cloud needs >= 2 points)
    train_eval: DataLoader  # same images, test-time transforms (Probe Head features)
    val: DataLoader
    test: DataLoader
    num_classes: int


def make_loaders(
    dataset: str,
    normalization,
    root: str = "data",
    batch_size: int = 64,
    val_size: int = 5000,
    num_workers: int = 4,
    fake_size: int = 256,
) -> Loaders:
    if dataset not in NUM_CLASSES:
        raise ValueError(f"Unknown dataset {dataset!r}; known: {sorted(NUM_CLASSES)}.")
    normalize = transforms.Normalize(*normalization)
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    augmented = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    def build(train: bool, transform):
        if dataset == "fake":
            n = fake_size if train else fake_size // 2
            return datasets.FakeData(
                size=n,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
                random_offset=0 if train else n,
            )
        return datasets.CIFAR10(root, train=train, transform=transform, download=True)

    train_aug, train_plain, test = build(True, augmented), build(True, plain), build(False, plain)
    n = len(train_aug)
    val_size = min(val_size, n // 5)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(VAL_SPLIT_SEED)).tolist()
    val_idx, train_idx = order[:val_size], order[val_size:]

    def loader(ds, shuffle=False, drop_last=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return Loaders(
        train=loader(Subset(train_aug, train_idx), shuffle=True, drop_last=True),
        train_eval=loader(Subset(train_plain, train_idx)),
        val=loader(Subset(train_plain, val_idx)),
        test=loader(test),
        num_classes=NUM_CLASSES[dataset],
    )
