"""Run the experiment grid, one run after another; re-running resumes where it stopped.

    uv run python -m topkd_imat.training.grid --inits pretrained --skip-top --epochs 30

Per Initialisation: label-supervised x {Baseline, KD, Top, KD+Top} and label-free x
{KD, Top, KD+Top} (Baseline needs labels) = 7 runs. ``--skip-top`` leaves out the
Conditions that need the Topology Distillation Loss. Every other option (``--epochs``,
``--w-top``, ``--runs-dir``, ...) is passed to ``topkd_imat.training.run`` unchanged.
"""

import argparse
import sys
import traceback

from topkd_imat.training import run
from topkd_imat.training.results import CONDITIONS, INITS, has_row, run_name

LABEL_FREE = ("KD", "Top", "KD+Top")


def plan_runs(inits=("pretrained",), skip_top=False, seeds=(0,), dataset="CIFAR-10") -> list[dict]:
    runs = []
    for seed in seeds:
        for init in inits:
            for supervision, conditions in (
                ("label-supervised", CONDITIONS),
                ("label-free", LABEL_FREE),
            ):
                for condition in conditions:
                    if skip_top and "Top" in condition:
                        continue
                    runs.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "supervision": supervision,
                            "init": init,
                            "seed": seed,
                        }
                    )
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--inits", nargs="+", choices=INITS, default=["pretrained"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--skip-top", action="store_true", help="only Baseline and KD")
    p.add_argument("--dataset", default="CIFAR-10")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    args, passthrough = p.parse_known_args(argv)

    runs = plan_runs(args.inits, args.skip_top, args.seeds, args.dataset)
    results = f"{args.runs_dir}/results.csv"
    todo = [r for r in runs if not has_row(results, r)]
    print(f"Grid: {len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to go.")
    for r in runs:
        print(f"  [{'done' if r not in todo else 'todo'}] {run_name(r)}")
    if args.dry_run:
        return 0

    failed = []
    for i, r in enumerate(todo, 1):
        print(f"\n=== {i}/{len(todo)}: {run_name(r)} ===", flush=True)
        argv_run = [
            "--condition", r["condition"],
            "--supervision", r["supervision"],
            "--init", r["init"],
            "--seed", str(r["seed"]),
            "--dataset", r["dataset"],
            "--runs-dir", args.runs_dir,
            *passthrough,
        ]  # fmt: skip
        try:
            run.main(argv_run)
        except Exception:  # keep going: one broken run must not waste the night
            traceback.print_exc()
            failed.append(run_name(r))
    if failed:
        print(
            f"\n{len(failed)} run(s) failed (re-run the grid to retry them):", *failed, sep="\n  "
        )
        return 1
    print("\nGrid finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
