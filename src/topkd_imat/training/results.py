"""The Results table (docs/interfaces.md #3): one CSV row per training run.

``append_row`` validates a row before writing it, so a bad value fails loudly at the end
of a run instead of surfacing later in Story's plots.
"""

import csv
import math
from pathlib import Path

CONTRACT = "docs/interfaces.md #3"

COLUMNS = (
    "dataset",
    "condition",
    "supervision",
    "init",
    "seed",
    "accuracy",
    "topology_distance",
)
CONDITIONS = ("Baseline", "KD", "Top", "KD+Top")
SUPERVISIONS = ("label-supervised", "label-free")
INITS = ("scratch", "pretrained")
KEY = ("dataset", "condition", "supervision", "init", "seed")


class ResultsError(ValueError):
    """Raised when a Results row breaks the contract."""


def uses_probe_head(condition: str, supervision: str) -> bool:
    """Only Top without labels leaves the Student's classifier untrained."""
    return condition == "Top" and supervision == "label-free"


def run_name(row: dict) -> str:
    """Directory name of a run's artifacts, next to the results table."""
    return "{dataset}_{condition}_{supervision}_{init}_s{seed}".format(**row)


def validate_row(row: dict) -> dict:
    """Return the row with normalised types, or raise ResultsError saying what is wrong."""
    missing = [c for c in COLUMNS if c not in row]
    extra = [c for c in row if c not in COLUMNS]
    if missing or extra:
        raise ResultsError(
            f"Results row must have exactly the columns {list(COLUMNS)}; "
            f"missing {missing}, unexpected {extra} ({CONTRACT})."
        )
    for col, allowed in (("condition", CONDITIONS), ("supervision", SUPERVISIONS), ("init", INITS)):
        if row[col] not in allowed:
            raise ResultsError(f"{col} must be one of {list(allowed)}, got {row[col]!r}.")
    if row["condition"] == "Baseline" and row["supervision"] == "label-free":
        raise ResultsError("Baseline trains on labels only, so it cannot be label-free.")
    if not str(row["dataset"]).strip():
        raise ResultsError("dataset must be a non-empty name.")
    try:
        seed = int(row["seed"])
        accuracy = float(row["accuracy"])
        distance = float(row["topology_distance"])
    except (TypeError, ValueError) as err:
        raise ResultsError(f"seed must be an int and the metrics numbers: {err}") from err
    if not 0.0 <= accuracy <= 1.0:
        raise ResultsError(f"accuracy is a fraction in [0, 1], got {accuracy}.")
    if not (math.isnan(distance) or distance >= 0.0):
        raise ResultsError(
            f"topology_distance must be >= 0 or NaN (not computed yet), got {distance}."
        )
    return {**row, "seed": seed, "accuracy": accuracy, "topology_distance": distance}


def read_rows(path: Path) -> list[dict]:
    """All rows of a results table (empty if the file does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ResultsError(
                f"{path} has columns {reader.fieldnames}, expected {list(COLUMNS)} ({CONTRACT})."
            )
        return [validate_row(r) for r in reader]


def has_row(path: Path, key: dict) -> bool:
    """True if a row with the same dataset/condition/supervision/init/seed exists."""
    want = tuple(str(key[k]) for k in KEY)
    return any(tuple(str(r[k]) for k in KEY) == want for r in read_rows(path))


def append_row(path: Path, row: dict) -> None:
    """Validate ``row`` and append it, writing the header if the file is new."""
    row = validate_row(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole table (used to fill topology_distance afterwards)."""
    rows = [validate_row(r) for r in rows]
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
