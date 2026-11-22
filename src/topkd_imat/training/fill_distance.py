"""Fill the ``topology_distance`` column from the saved test-set clouds.

    uv run python -m topkd_imat.training.fill_distance --runs-dir runs

Runs record NaN while training; once the Topology subteam's ``topology_distance`` exists,
this computes it at the last Layer Pair for every row still missing it. Rows are only
rewritten if every distance could be computed.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from topkd_imat.training.results import read_rows, run_name, write_rows


def last_pair(path: Path) -> np.ndarray:
    with np.load(path) as clouds:
        pairs = sorted((k for k in clouds.files if k.startswith("pair_")), key=_pair_index)
        return clouds[pairs[-1]].astype(np.float32)


def _pair_index(name: str) -> int:
    return int(name.split("_")[1])


def fill(runs_dir: Path, distance_fn=None) -> int:
    """Fill missing distances in ``runs_dir/results.csv``; return how many were filled."""
    if distance_fn is None:
        from topkd_imat.topology import topology_distance as distance_fn
    results = runs_dir / "results.csv"
    rows = read_rows(results)
    missing = [r for r in rows if math.isnan(r["topology_distance"])]
    if not missing:
        return 0
    teacher = last_pair(runs_dir / "teacher" / "clouds_test.npz")
    for r in missing:
        student = last_pair(runs_dir / run_name(r) / "clouds_test.npz")
        r["topology_distance"] = float(distance_fn(teacher, student))
    write_rows(results, rows)
    return len(missing)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)
    try:
        n = fill(Path(args.runs_dir))
    except NotImplementedError as err:
        print(f"Nothing changed: {err}")
        return 1
    print(f"Filled topology_distance for {n} run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
