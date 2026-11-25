"""Which Layer Pairs carry topology worth distilling? The evidence behind ADR 0007.

    uv run --group analysis python scripts/analyse_layer_topology.py \
        --root results --out results/layer_topology

Reads the test-set Latent Point Clouds a training run saves (``<root>/<run>/*/clouds_test.npz``
and the Teacher's ``<root>/<run>/teacher/clouds_test.npz``), so it needs the runs on disk; the
clouds are not in git. Writes three CSVs to ``--out``:

1. ``h0_loss.csv``: the ``h0-pi-norm`` and ``h0-matched`` terms between Teacher and Student
   test clouds at each Layer Pair, in batches of 64 as in training (``h0-pi-norm`` with the
   Teacher calibration of a KD+Top run's config).
2. ``h0_class_structure.csv``: what the H0 diagram says about the classes. Normalised MST deaths
   (quantiles; the 9 longest against the rest: 10 Class Clusters show as 9 long edges), the
   mean distance to the class centroid over the mean distance between centroids, the accuracy
   of the nearest class centroid, and the share of MST edges that join two images of the same
   class (short edges) or of different classes (the 9 longest).
3. ``higher_homology.csv``: H1..Hk of Vietoris-Rips on normalised distances, for random subsets
   of the test cloud, against two nulls of the same size: one Gaussian with the cloud's mean
   and covariance, and a mixture of one Gaussian per class (each class's own mean and
   covariance, same labels). A real feature should be more persistent than the nulls' most
   persistent one; ``share_above_*`` is the share of subsets where it is.

``ripser`` is only needed for 3 (``--group analysis``).
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist, squareform

from topkd_imat.topology.h0 import h0_deaths
from topkd_imat.topology.matched import H0MatchedLoss
from topkd_imat.topology.paper_pi import PaperH0PILoss, PaperPICalibration
from topkd_imat.training.loop import topology_between

DEFAULT_RUNS = ["KD_wkd1", "KD_wkd2", "h0-pi-norm_wkd1_wtop3", "h0-pi-norm_wkd1_wtop10",
                "h0-pi-norm_wkd2_wtop3", "h0-pi-norm_wkd2_wtop10", "h0-pi-norm_wkd1_wtop30",
                "h0-pi-norm_wkd1_wtop100"]  # fmt: skip
HOMOLOGY_RUNS = ["KD_wkd1", "h0-pi-norm_wkd1_wtop3", "h0-pi-norm_wkd1_wtop10"]  # Rips is slow
BATCH = 64
N_CLASSES = 10


def load_clouds(root: Path, runs: list[str]) -> dict[str, dict]:
    out = {"Teacher": dict(np.load(root / runs[0] / "teacher" / "clouds_test.npz"))}
    for run in runs:
        found = glob.glob(str(root / run / "*" / "clouds_test.npz"))
        (path,) = [q for q in found if Path(q).parent.name != "teacher"]
        out[run] = dict(np.load(path))
    return out


def calibration(root: Path, runs: list[str]) -> list[PaperPICalibration]:
    for run in runs:
        for path in glob.glob(str(root / run / "*" / "config.json")):
            cal = json.loads(Path(path).read_text()).get("pi_calibration")
            if cal:
                return [PaperPICalibration(**c) for c in cal]
    raise SystemExit("No run with a pi_calibration in its config.json (a KD+Top h0-pi-norm run).")


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            [{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for r in rows]
        )
    print(f"wrote {path}")


def h0_loss(clouds: dict, cal: list[PaperPICalibration]) -> list[dict]:
    losses = [PaperH0PILoss(c) for c in cal]
    teacher = [
        torch.from_numpy(clouds["Teacher"][f"pair_{p}"].astype(np.float32)) for p in range(3)
    ]
    rows = []
    for run, z in clouds.items():
        if run == "Teacher":
            continue
        student = [torch.from_numpy(z[f"pair_{p}"].astype(np.float32)) for p in range(3)]
        pi = topology_between(losses, teacher, student, BATCH)
        matched = topology_between(H0MatchedLoss(), teacher, student, BATCH)
        rows += [
            {"run": run, "pair": p, "h0_pi_norm": pi[f"top_{p}"], "h0_matched": matched[f"top_{p}"]}
            for p in range(3)
        ]
    return rows


def h0_class_structure(clouds: dict) -> list[dict]:
    rows = []
    for run, z in clouds.items():
        y = z["labels"]
        for p in range(3):
            x = z[f"pair_{p}"].astype(np.float64)
            deaths, same_short, cross_long = [], [], []
            for s in range(0, len(x) - BATCH + 1, BATCH):
                xb, yb = x[s : s + BATCH], y[s : s + BATCH]
                deaths.append(np.sort(h0_deaths(torch.from_numpy(xb)).numpy()))
                tree = minimum_spanning_tree(cdist(xb, xb)).tocoo()
                order = np.argsort(tree.data)
                same = yb[tree.row[order]] == yb[tree.col[order]]
                k = len(np.unique(yb)) - 1  # edges that must join different classes
                same_short.append(same[:-k].mean())
                cross_long.append(1 - same[-k:].mean())
            deaths = np.array(deaths)
            mean_sorted = deaths.mean(0)
            xs = x / np.linalg.norm(x - x.mean(0), axis=1).mean()
            cents = np.stack([xs[y == c].mean(0) for c in range(N_CLASSES)])
            intra = np.mean([np.linalg.norm(xs[y == c] - cents[c], axis=1).mean()
                             for c in range(N_CLASSES)])  # fmt: skip
            inter = pdist(cents).mean()
            nearest = np.argmin(cdist(xs, cents), axis=1)
            rows.append(
                {
                    "run": run,
                    "pair": p,
                    "death_q10": float(np.quantile(deaths, 0.1)),
                    "death_q50": float(np.median(deaths)),
                    "death_q90": float(np.quantile(deaths, 0.9)),
                    "longest9_over_rest": float(mean_sorted[-9:].mean() / mean_sorted[:-9].mean()),
                    "intra_over_inter": float(intra / inter),
                    "nearest_centroid_acc": float((nearest == y).mean()),
                    "mst_short_edges_same_class": float(np.mean(same_short)),
                    "mst_longest_edges_cross_class": float(np.mean(cross_long)),
                }
            )
    return rows


def higher_homology(clouds: dict, runs: list[str], sizes: list[tuple[int, int, int]]) -> list:
    from ripser import ripser

    def persistences(x, maxdim):
        d = squareform(pdist(x))
        d /= d[np.triu_indices(len(x), 1)].mean()
        out = []
        for dgm in ripser(d, distance_matrix=True, maxdim=maxdim)["dgms"]:
            p = dgm[:, 1] - dgm[:, 0] if len(dgm) else np.zeros(0)
            out.append(p[np.isfinite(p)])
        return out

    def mixture(x, y, ii, rng):
        out = np.empty((len(ii), x.shape[1]))
        for c in np.unique(y[ii]):
            sel = y[ii] == c
            xc = x[y == c]
            out[sel] = rng.multivariate_normal(xc.mean(0), np.cov(xc.T), sel.sum())
        return out

    peak = lambda p: float(p.max()) if len(p) else 0.0  # noqa: E731
    rows = []
    for n, maxdim, reps in sizes:
        for p in range(3):
            for run in ["Teacher", *runs]:
                rng = np.random.default_rng(0)  # same subsets for every run
                subsets = [rng.choice(10_000, n, replace=False) for _ in range(reps)]
                x = clouds[run][f"pair_{p}"].astype(np.float64)
                y = clouds[run]["labels"]
                mu, cov = x.mean(0), np.cov(x.T)
                stats = {k: [] for k in range(1, maxdim + 1)}
                for ii in subsets:
                    real = persistences(x[ii], maxdim)
                    gauss = persistences(rng.multivariate_normal(mu, cov, n), maxdim)
                    mix = persistences(mixture(x, y, ii, rng), maxdim)
                    for k in stats:
                        stats[k].append((len(real[k]), peak(real[k]), len(gauss[k]),
                                         peak(gauss[k]), len(mix[k]), peak(mix[k])))  # fmt: skip
                for k, v in stats.items():
                    v = np.array(v)
                    rows.append(
                        {
                            "n_points": n, "pair": p, "run": run, "dim": k, "subsets": reps,
                            "count": v[:, 0].mean(), "max_persistence": v[:, 1].mean(),
                            "gauss_count": v[:, 2].mean(), "gauss_max": v[:, 3].mean(),
                            "mixture_count": v[:, 4].mean(), "mixture_max": v[:, 5].mean(),
                            "share_above_gauss": float((v[:, 1] > v[:, 3]).mean()),
                            "share_above_mixture": float((v[:, 1] > v[:, 5]).mean()),
                        }
                    )  # fmt: skip
                print(f"  H1..H{maxdim}, {n} points, pair {p}: {run}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", default="results", help="folder with one sub-folder per run")
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--homology-runs", nargs="+", default=HOMOLOGY_RUNS, help="part 3 only")
    p.add_argument("--out", default="results/layer_topology")
    p.add_argument(
        "--homology",
        nargs="*",
        default=["64:4:20", "150:3:5"],
        help="points:max_dim:subsets for each Rips computation (none: skip part 3)",
    )
    args = p.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clouds = load_clouds(root, args.runs)
    write(out / "h0_loss.csv", h0_loss(clouds, calibration(root, args.runs)))
    write(out / "h0_class_structure.csv", h0_class_structure(clouds))
    if args.homology:
        sizes = [tuple(int(v) for v in s.split(":")) for s in args.homology]
        write(out / "higher_homology.csv", higher_homology(clouds, args.homology_runs, sizes))


if __name__ == "__main__":
    main()
