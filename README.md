# TopKD-imat

Trabajo final — Geometría de la Información (MIA).

We apply **TopKD** — knowledge distillation that also transfers the *topology* of the Teacher's latent space (Kim et al., ICML 2024, [paper](https://proceedings.mlr.press/v235/kim24aj.html), [code](https://github.com/jekim5418/TopKD)) — to a small image-classification task (CIFAR-10), computing persistence images exactly with the course's TDA libraries instead of approximating them with RipsNet, so we can *see* whether the Student's latent space takes on the Teacher's shape.

## Start here

| Read | For |
|---|---|
| [`docs/plan.md`](docs/plan.md) | Research question, experiment grid, **workstreams to pick**, subteams, memoria split, team rules |
| [`CONTEXT.md`](CONTEXT.md) | The shared vocabulary (English + the Spanish wording for the memoria) |
| [`docs/interfaces.md`](docs/interfaces.md) | The three contracts between subteams |
| [`docs/adr/`](docs/adr/) | Decisions and why we made them |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `make`.

```bash
make install   # uv sync: creates .venv from uv.lock
make test      # must pass before anything reaches main
```

On Windows without WSL, install `make` (e.g. `winget install GnuWin32.Make`) or run the commands from the `Makefile` directly (`uv sync`, `uv run pytest`).

Adding a dependency: `uv add <package>` (commit `pyproject.toml` and `uv.lock` together).

## Repository layout

```
src/topkd_imat/
  data/        Models subteam       dataset loaders
  models/      Models subteam       Teacher, Student            -> Model interface
  topology/    Topology subteam     PD / exact PI / loss        -> Loss interface
  training/    Experiments subteam  Conditions, grid, Probe Head -> Results interface
  analysis/    Story subteam        figures and tables (reads result files only)
tests/
  contracts/   contract tests + stubs, written by each interface's consumer
scripts/       entry points
notebooks/     exploration only — never imported
results/       small results (CSV, figures); large artifacts go to DagsHub / HF
report/        memoria (format to be decided)
slides/        presentation (format to be decided)
docs/          plan, interfaces, ADRs
```

## Rules (short version — full list in [`docs/plan.md`](docs/plan.md#team-rules))

1. **`main` always runs** (`make test` passes). Work on a branch; merge into `main` only when the tests pass.
2. **Push with `make push`** — it runs the tests first. A GitHub Action re-runs them on every push.
3. **Stay in your subteam's folder**; cross folders only through [`docs/interfaces.md`](docs/interfaces.md).
4. **The consumer writes the contract tests** for what it consumes, plus a stub.
5. **No datasets, checkpoints or point clouds in git.**
6. `make lint` before pushing is appreciated, not required.
7. The memoria is written by us, in Spanish, by the owner of each part — generative AI is not allowed for writing it.
