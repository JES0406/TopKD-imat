"""Sanity checks for the Teacher and the pretrained Student (W1, W2).

    uv run python scripts/evaluate_models.py [--skip-cifar100] [--per-class 20]

1. Teacher test accuracy on CIFAR-10 with INPUT_NORMALIZATION, and with the "true" CIFAR-10
   std instead, to show what a wrong normalisation costs.
2. The CIFAR-100 ResNet-20 the Student comes from: CIFAR-100 test accuracy before and after
   depth-pruning to ResNet-8 (no retraining), i.e. how much the pruning itself destroys.
3. The Student as handed to training (pruned + fresh 10-class head) on CIFAR-10: chance level.
4. H0 of the Teacher's last Latent Point Cloud on a class-balanced test batch. For a Rips
   filtration the H0 death times are the edge lengths of the Euclidean minimum spanning tree,
   so no TDA library is needed: 10 Class Clusters show up as 9 edges much longer than the rest.

Reads models built in memory (same code as scripts/build_models.py); downloads datasets to data/.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from topkd_imat import models
from topkd_imat.models.build import (
    UPSTREAM_NORMALIZATION,
    build_student,
    build_teacher,
    load_upstream,
)
from topkd_imat.models.resnet_cifar import prune_depth

TRUE_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def test_set(name: str, normalization, root: str = "data", train: bool = False):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalization)])
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    return cls(root, train=train, transform=tf, download=True)


@torch.no_grad()
def recalibrate_bn(model, dataset, n_images: int = 10_000) -> None:
    """Recompute BatchNorm running statistics on ``dataset`` images (no labels, no gradient).

    Removing blocks changes the activations every later BatchNorm sees, so its stored
    statistics no longer match; this separates that mismatch from knowledge actually lost.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all batches
    model.train()
    sub = torch.utils.data.Subset(dataset, range(n_images))
    for x, _ in DataLoader(sub, batch_size=500):
        model(x)
    model.eval()


@torch.no_grad()
def accuracy(model, dataset, batch_size: int = 500) -> float:
    model.eval()
    correct = 0
    for x, y in DataLoader(dataset, batch_size=batch_size):
        correct += int((model(x)[0].argmax(1) == y).sum())
    return correct / len(dataset)


def mst_edge_lengths(points: np.ndarray) -> np.ndarray:
    """Prim's algorithm on the complete Euclidean graph; returns the n-1 MST edge lengths."""
    n = len(points)
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    parent = np.zeros(n, dtype=int)
    edges = []
    for _ in range(n - 1):
        best_masked = np.where(in_tree, np.inf, best)
        j = int(best_masked.argmin())
        edges.append((parent[j], j, best_masked[j]))
        in_tree[j] = True
        closer = dist[j] < best
        parent[closer & ~in_tree] = j
        best = np.minimum(best, dist[j])
    return edges


def h0_report(cloud: np.ndarray, labels: np.ndarray, n_clusters: int = 10) -> dict:
    edges = mst_edge_lengths(cloud)
    lengths = np.sort([e[2] for e in edges])[::-1]
    k = n_clusters - 1
    # Cut the k longest MST edges -> n_clusters components (single linkage); measure purity.
    keep = sorted(edges, key=lambda e: e[2])[: len(edges) - k]
    root = list(range(len(cloud)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for a, b, _ in keep:
        root[find(a)] = find(b)
    comps = np.array([find(i) for i in range(len(cloud))])
    majority = sum(np.bincount(labels[comps == c]).max() for c in np.unique(comps))
    return {
        "top_bars": lengths[: k + 3],
        "median_bar": float(np.median(lengths)),
        "gap_ratio": float(lengths[k - 1] / lengths[k]),  # 9th longest / 10th longest
        "component_sizes": sorted(np.bincount(np.unique(comps, return_inverse=True)[1]))[::-1],
        "purity": majority / len(cloud),
    }


@torch.no_grad()
def balanced_cloud(model, dataset, per_class: int, layer: int = -1, seed: int = 0):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(targets == c), per_class, replace=False) for c in range(10)]
    )
    x = torch.stack([dataset[i][0] for i in idx])
    model.eval()
    return model(x)[1][layer].numpy(), targets[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--skip-cifar100", action="store_true")
    p.add_argument("--per-class", type=int, nargs="+", default=[10, 20])
    args = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    teacher, student = build_teacher(progress=False), build_student(progress=False)
    c10 = test_set("cifar10", models.INPUT_NORMALIZATION)

    print("== 1. Teacher (ResNet-56, CIFAR-10) test accuracy")
    print(
        f"  with INPUT_NORMALIZATION {models.INPUT_NORMALIZATION[1]}: {accuracy(teacher, c10):.4f}"
    )
    wrong = test_set("cifar10", (models.INPUT_NORMALIZATION[0], TRUE_CIFAR10_STD))
    print(f"  with the true CIFAR-10 std {TRUE_CIFAR10_STD}:   {accuracy(teacher, wrong):.4f}")

    if not args.skip_cifar100:
        print("== 2. Source of the Student: chenyaofo ResNet-20 on CIFAR-100 (its own test set)")
        c100 = test_set("cifar100", UPSTREAM_NORMALIZATION["cifar100"])
        r20 = load_upstream("cifar100", 20, progress=False)
        print(f"  ResNet-20 (as released):              {accuracy(r20, c100):.4f}")
        pruned = prune_depth(r20, 1)
        print(f"  pruned to ResNet-8, no retraining:    {accuracy(pruned, c100):.4f}")
        train100 = test_set("cifar100", UPSTREAM_NORMALIZATION["cifar100"], train=True)
        recalibrate_bn(pruned, train100)
        print(f"  ... + BatchNorm stats recomputed:     {accuracy(pruned, c100):.4f}")

    print("== 3. Student handed to training (pruned + fresh 10-class head) on CIFAR-10")
    print(f"  test accuracy: {accuracy(student, c10):.4f}  (chance = 0.10)")

    print("== 4. H0 of the Teacher's last Latent Point Cloud (class-balanced test batches)")
    for per_class in args.per_class:
        for name, model in (("Teacher", teacher), ("Student before training", student)):
            cloud, labels = balanced_cloud(model, c10, per_class)
            r = h0_report(cloud, labels)
            bars = ", ".join(f"{v:.2f}" for v in r["top_bars"])
            print(
                f"  {name}, {10 * per_class} points: longest finite H0 bars [{bars}] "
                f"(median {r['median_bar']:.2f}); 9th/10th = {r['gap_ratio']:.2f}; "
                f"cutting 9 longest -> sizes {r['component_sizes']}, purity {r['purity']:.2f}"
            )


if __name__ == "__main__":
    main()
