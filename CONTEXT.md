# TopKD-imat

Final project for Geometría de la Información (GI, MIA): applying the TopKD technique — knowledge distillation that also transfers the topology of a network's latent space — to a simpler, more interpretable image task, with persistence images computed exactly instead of approximated by RipsNet.

Code, docs and this glossary are in English; the written report (memoria) is in Spanish. Each term lists the Spanish wording to use in the memoria and slides, so that all authors write the same words.

## Language

### Distillation

**Teacher**:
The larger, pre-trained network whose knowledge is transferred.
_Español_: modelo profesor
_Avoid_: big model, source model; maestro, modelo docente

**Student**:
The smaller network trained to imitate the Teacher.
_Español_: modelo estudiante
_Avoid_: small model, target model; alumno

**Knowledge Distillation (KD)**:
Training a Student to match the Teacher's softened output distribution, in addition to the true labels.
_Español_: destilación de conocimiento
_Avoid_: model compression (broader), teacher-student training

**TopKD**:
The technique from Kim et al., ICML 2024 ("Do Topological Characteristics Help in Knowledge Distillation?"): KD plus a Topology Distillation Loss.
_Español_: TopKD
_Avoid_: TDA-KD, topological KD (as a name)

**Topology Distillation Loss**:
The loss term that penalises the difference between the Teacher's and the Student's Persistence Images for the same batch.
_Español_: pérdida de destilación topológica
_Avoid_: TDA loss, topo loss

### Topology

**Latent Point Cloud**:
The set of embedding vectors that one network produces, at a chosen layer, for the images in one batch — one point per image.
_Español_: nube de puntos del estado latente
_Avoid_: PCD, feature cloud, embeddings (when the batch-as-a-set meaning is intended)

**Persistence Diagram (PD)**:
The multiset of (birth, death) pairs of the homology features of a Latent Point Cloud's filtration.
_Español_: diagrama de persistencia
_Avoid_: barcode (same information, different picture — say "barcode" only for the plot)

**Persistence Image (PI)**:
A fixed-size pixel grid obtained by smoothing a Persistence Diagram, in (birth, persistence) coordinates, with persistence-weighted Gaussian kernels — as defined in the course (GI Tema 1, Topología, pp. 50–51).
_Español_: imagen de persistencia
_Avoid_: topological image, PD image

**Exact PI**:
A Persistence Image computed from a true Persistence Diagram, as opposed to one predicted by a network.
_Español_: imagen de persistencia exacta
_Avoid_: ground-truth PI, real PI

**RipsNet**:
The neural network TopKD uses to approximate a Persistence Image from a Latent Point Cloud. Out of scope for this project.
_Español_: RipsNet
_Avoid_: PI approximator

**Layer Pair**:
A Teacher layer and a Student layer whose Latent Point Clouds are compared by the Topology Distillation Loss.
_Español_: par de capas
_Avoid_: tap, matching layer, distillation point

**Class Cluster**:
The group of points in a Latent Point Cloud that belong to one digit class; at the last Layer Pair a good network shows one per class.
_Español_: cluster de la clase
_Avoid_: blob, digit cluster; cúmulo

### Experiment

**Core Dataset**:
MNIST — the dataset every Condition is run on and the one shown in the presentation.
_Español_: conjunto principal
_Avoid_: main dataset, base dataset

**Extension Dataset**:
A harder dataset of similar format (e.g. Fashion-MNIST) run as extra experiments for the written report only.
_Español_: conjuntos extra
_Avoid_: secondary dataset, hard dataset

**Condition**:
The combination of losses a Student is trained with. One of: **Baseline** (no Teacher), **KD**, **Top** (Topology Distillation Loss only), **KD+Top** (what the paper calls TopKD).
_Español_: condición — the four names stay in English as labels in tables and figures
_Avoid_: NoKD, VanillaKD, "TopKD" for the Top-only condition

**Label-supervised / Label-free**:
Whether the Student's training also uses the true labels (cross-entropy) or only signals from the Teacher.
_Español_: con etiquetas / sin etiquetas
_Avoid_: output-supervised, KD-only supervised

**Probe Head**:
A classification head trained with labels on top of a Student that was trained Label-free, used to measure how much class information its latent space holds.
_Español_: cabeza de sondeo
_Avoid_: linear probe (unless it is literally linear), classifier
