"""Search ``w_kd`` and ``w_top`` for KD+Top x label-supervised x scratch, last Layer Pair.

    uv run python -m topkd_imat.training.search --root runs_search --epochs 60 --seeds 0
    uv run python -m topkd_imat.training.search --root runs_search --summary-only

For each seed and each ``w_kd``: a KD control (``<root>/KD_wkd<k>/``) and, for each
``--top-losses`` (``h0-pi-norm`` and ``h0-matched`` by default), one KD+Top run per ``w_top``
(``<root>/<loss>-last_wkd<k>_wtop<w>/``), with the topology term at the last Layer Pair only
(``--top-pairs last``, ADR 0007). The control gets the same ``w_kd`` values, so KD+Top is
compared with a KD that had the same tuning, and both losses share it. Every run has its own
folder: run names do not include ``--w-kd``, ``--w-top``, ``--top-loss`` or ``--top-pairs``,
so sharing a folder would skip runs. Re-running resumes: a run whose row is already in its
folder's ``results.csv`` is skipped.

Defaults: ``w_top`` per loss (``W_TOPS``; ``--w-tops`` overrides it for every loss).

- ``h0-pi-norm`` 0.3 / 1 / 3. A first pass trained the mean over all three Layer Pairs
  (seed 0, 60 epochs, ``w_kd`` 1; records in ``results/search_h0-pi-norm_all-pairs/``):
  ``w_top`` 3 / 10 / 30 / 100 gave validation accuracy 86.4 / 85.9 / 82.9 / 57.9 % against
  KD's 86.5 %. That mean gave the last pair ``w_top / 3``; trained alone it gets ``w_top``,
  so 0.3 / 1 / 3 put the same weight on it as 1 / 3 / 10 did.
- ``h0-matched`` 1 / 3 / 10. On a scratch Student's first CIFAR-10 batches (last pair) its
  gradient on the Student's cloud is 0.4 times ``h0-pi-norm``'s at the same ``w_top`` (0.073
  against 0.18; KD's is 0.16), so about 2.5 times ``h0-pi-norm``'s weights give the same push.

``w_kd`` 1 / 2 / 4 brackets the paper's best (2; Appendix D.1, where it degrades from 7). The
KD temperature (4), the batch size (64) and the learning rate (0.1 from scratch) are not
searched: they are the paper's or the grid's recipe for every Condition, and the batch is also
the size of the Latent Point Cloud.

Every other option (``--device``, ``--num-workers``, ``--data-root``, ...) is passed to
``topkd_imat.training.run`` unchanged.

Criterion: the setting with the **highest validation accuracy at the last epoch** (the 5k
split held out of training; ``metrics.csv``), separately for KD (over ``w_kd``) and for each
loss (over ``w_kd`` x ``w_top``). Not the W4 sweep's "largest ``w_top`` that does not fall
below KD", and never the test accuracy in ``results.csv``: choosing on the test set would leak
it into the result. ``<root>/summary.csv`` lists every run, best first; ``val_top_last`` is
each run's own loss (KD logs ``run``'s default, ``h0-pi-norm``), so compare it within a loss.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

from topkd_imat.training import run

TOP_PAIRS = "last"
SUPERVISION, INIT = "label-supervised", "scratch"
W_TOPS = {"h0-pi-norm": (0.3, 1.0, 3.0), "h0-matched": (1.0, 3.0, 10.0)}
W_KDS = (1.0, 2.0, 4.0)
SET_BY_SEARCH = (
    "--condition", "--supervision", "--init", "--seed", "--runs-dir", "--top-loss", "--w-top",
    "--w-kd", "--top-pairs",
)  # fmt: skip
SUMMARY_FIELDS = [
    "config", "condition", "top_loss", "w_kd", "w_top", "seed", "epochs_done", "val_acc_last",
    "val_acc_best", "val_top_last", "run_dir",
]  # fmt: skip


def plan(w_kds, w_tops: dict, seeds) -> list[dict]:
    """Every run: for each seed and w_kd, the KD control and KD+Top for each loss and w_top.

    ``w_tops`` maps each loss to its ``w_top`` values.
    """
    runs = []
    for seed in seeds:
        for k in w_kds:
            runs.append(
                {"config": f"KD_wkd{k:g}", "condition": "KD", "top_loss": None, "w_kd": k,
                 "w_top": None, "seed": seed}
            )  # fmt: skip
            for loss, values in w_tops.items():
                for w in values:
                    runs.append(
                        {"config": f"{loss}-{TOP_PAIRS}_wkd{k:g}_wtop{w:g}",
                         "condition": "KD+Top", "top_loss": loss, "w_kd": k, "w_top": w,
                         "seed": seed}
                    )  # fmt: skip
    return runs


def run_argv(r: dict, root: Path, passthrough: list[str]) -> list[str]:
    argv = [
        "--condition", r["condition"],
        "--supervision", SUPERVISION,
        "--init", INIT,
        "--seed", str(r["seed"]),
        "--w-kd", f"{r['w_kd']:g}",
        "--runs-dir", str(root / r["config"]),
    ]  # fmt: skip
    if r["w_top"] is not None:
        argv += ["--top-loss", r["top_loss"], "--top-pairs", TOP_PAIRS]
        argv += ["--w-top", f"{r['w_top']:g}"]
    return [*argv, *passthrough]


def _metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarise(root: Path, runs: list[dict]) -> list[dict]:
    """One row per planned run that has metrics, best last-epoch validation accuracy first."""
    rows = []
    for r in runs:
        pattern = f"*_{r['condition']}_{SUPERVISION}_{INIT}_s{r['seed']}"  # any --dataset
        dirs = sorted(p for p in (root / r["config"]).glob(pattern) if p.is_dir())
        metrics = _metrics(dirs[0]) if dirs else []
        if not metrics:
            continue
        val = [float(m["val_acc"]) for m in metrics]
        rows.append(
            {
                **{k: r[k] for k in ("config", "condition", "top_loss", "w_kd", "w_top", "seed")},
                "epochs_done": len(metrics),
                "val_acc_last": val[-1],
                "val_acc_best": max(val),
                "val_top_last": float(metrics[-1]["val_top"]),
                "run_dir": str(dirs[0]),
            }
        )
    rows.sort(key=lambda row: -row["val_acc_last"])
    return rows


def write_summary(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    # No abbreviations: --w-top / --w-kd / --seed would otherwise be read as --w-tops / ...
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--root", default="runs_search", help="one sub-folder per configuration")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--w-kds", nargs="+", type=float, default=W_KDS)
    p.add_argument("--top-losses", nargs="+", choices=list(W_TOPS), default=list(W_TOPS))
    p.add_argument("--w-tops", nargs="+", type=float, help="the same w_top values for every loss")
    p.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    p.add_argument("--summary-only", action="store_true", help="only rebuild summary.csv")
    args, passthrough = p.parse_known_args(argv)
    for flag in SET_BY_SEARCH:
        if any(a == flag or a.startswith(flag + "=") for a in passthrough):
            p.error(f"{flag} is set by the search; remove it")

    root = Path(args.root)
    w_tops = {loss: args.w_tops or W_TOPS[loss] for loss in args.top_losses}
    runs = plan(args.w_kds, w_tops, args.seeds)
    setting = f"{', '.join(w_tops)}; {TOP_PAIRS} pair, {SUPERVISION}, {INIT}"
    print(f"Search: {len(runs)} runs in {root}/ ({setting}).")
    for r in runs:
        print(f"  {r['config']}  seed {r['seed']}")
    if args.dry_run:
        return 0

    failed = []
    if not args.summary_only:
        for i, r in enumerate(runs, 1):
            print(f"\n=== {i}/{len(runs)}: {r['config']} seed {r['seed']} ===", flush=True)
            try:
                run.main(run_argv(r, root, passthrough))
            except Exception:  # keep going: one broken run must not waste the night
                traceback.print_exc()
                failed.append(f"{r['config']} seed {r['seed']}")

    rows = summarise(root, runs)
    path = write_summary(root, rows)
    print(f"\nValidation accuracy, last epoch ({path}):")
    for row in rows:
        print(
            f"  {row['val_acc_last']:.4f} (best {row['val_acc_best']:.4f})  "
            f"{row['config']}  seed {row['seed']}  [{row['epochs_done']} epochs]"
        )
    if failed:
        print(f"\n{len(failed)} run(s) failed (re-run to retry them):", *failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
