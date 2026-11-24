# H0 loss with correspondence: the Student keeps the Teacher's MST edges, image by image

*Status: proposed by the Topology owner (W3, Marcos) on 2026-11-24, as an extra loss next to ADR 0004's; whether its runs go into the results is W4's call once they exist.*

The final W4 grid (60 epochs, `w_top` 0.1, seeds 0–1) shows that ADR 0004's loss works as a loss: every Condition with Top roughly halves the `topology_distance` to the Teacher. But it does not improve accuracy, and Top × label-free stays at 18.6 % (*scratch*) and 38 % (*pretrained*) with the Probe Head. The reason is the one ADR 0004 lists as a limitation: a Persistence Diagram does not know which point is which, so the Student copies the scales at which points merge but not *which images* go together. H1 has the same blindness, so it is not the fix.

The Teacher's and the Student's clouds hold the same images in the same order, so an edge (i, j) of the Teacher's minimum spanning tree (MST) names a pair of images whose distance can be read in the Student. **`H0MatchedLoss`** (`topology/matched.py`) asks the two clouds to agree on the distances that define each other's H0:

L = ½ · [ mean over (i, j) in MST_T of (D_S[i, j] − D_T[i, j])² + mean over (i, j) in MST_S of (D_S[i, j] − D_T[i, j])² ]

with D each cloud's pairwise distances divided by their own mean, as in ADR 0004. The Teacher's edges keep the Teacher's merges: images that are neighbours in the Teacher stay neighbours, and clusters the Teacher separates stay apart through its long edges. The Student's edges stop the Student from creating short cuts the Teacher does not have. This is the topological loss of Moor et al., "Topological Autoencoders" (ICML 2020), restricted to H0 and used for distillation instead of between an input and its code. Gradients reach the Student as in ADR 0004: the MSTs are found on detached distances, and the loss reads `D_S[i, j]` from the distance matrix computed in torch.

**Toy check** (`tests/topology/test_matched.py`; 10 clusters of 8 points in 64-D as the Teacher, a random 16-D cloud as the Student, Adam, 300 steps, 5 seeds). After training, the Student's MST is cut at its 9 longest edges and each group is scored by its majority label. Purity is **0.21 with `H0PILoss`** on every seed, close to a random grouping, and **1.00 with `H0MatchedLoss`** on every seed. A Student that is the Teacher with its rows shuffled gets PI loss exactly 0 and matched loss 0.94.

## Consequences

- Choose the loss with `run --top-loss h0-matched` (the grid passes the flag through); the other option is `h0-pi-norm`, the default ([ADR 0006](0006-paper-pi.md)). `config.json` records the loss used. Use a separate `--runs-dir`: run names do not include the loss.
- **`w_top` must be retuned.** The matched loss is on a much smaller scale than the PI MSE: on the toy clouds, a random Student scores 43.7 with the PI loss and 0.56 with the matched loss, and a noisy copy of the Teacher 4.4 against 0.015. `w_top` 0.1 would make the topology term almost vanish; start with `w_top` 1 and compare at least two values.
- Cost is the same as `H0PILoss` (~2.5 ms per Layer Pair and batch of 64 on CPU): the same two MSTs, and no image to build.
- It is a deviation from the paper beyond ADR 0001's: the loss no longer compares Persistence Images and uses the pairing of images across networks, which TopKD does not. Report it in the memoria as our variant, with Moor et al. as its source.
- Unlike the PI loss, this one can transfer class information without labels, through which images are neighbours in the Teacher. Top × label-free with a Probe Head is the run that tests it.
- At the first two Layer Pairs it matches the Teacher's nearest-neighbour structure, which is stronger than matching distance distributions; if it hurts accuracy, restricting it to the last Layer Pair is the first thing to try.
- `topology_distance` (Wasserstein-2 between diagrams) is still blind to correspondence, so it will not show this loss's gain. The Probe Head accuracy will.
