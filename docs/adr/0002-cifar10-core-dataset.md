# CIFAR-10 as the Core Dataset instead of MNIST

At the team meeting of 2026-11-21 we chose CIFAR-10 as the Core Dataset, replacing MNIST. (Rationale below drafted after the meeting — correct it if the team's reasons differ.) CIFAR-10 keeps what made MNIST attractive — 10 classes, so a good Teacher's last Layer Pair should show 10 Class Clusters (9 long-lived H0 bars plus one infinite bar) that can be checked by eye — while being hard enough that a shallow Student does not reach the Teacher's accuracy on its own, which leaves room for KD and the Topology Distillation Loss to make a measurable difference. It is also closer to the paper's setting (CIFAR-100).

## Consequences

- The cost of an Exact PI depends on the batch size and the latent dimension, not on the input images, so ADR 0001 still holds; the extra cost is in the Teacher and Student forward passes and in training time (W4 budget, DGX scheduling).
- Teacher and Student must be convolutional networks suited to 32×32×3 inputs; pretrained CIFAR-10 weights are easier to find online than MNIST ones (W1, W2).
- Extension Datasets are no longer "harder MNIST-like" sets; which ones (if any) is W7's call — MNIST / Fashion-MNIST can now serve as an easy sanity check.
