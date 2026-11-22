"""Contract tests for the Results interface (docs/interfaces.md #3).

Written by: Experiments subteam (provider; agreed with Story, the consumer).
results_example.csv is a stub table with made-up numbers, so Story can build plots
before real runs exist. Every real results table must pass the same checks.
"""

import math
from pathlib import Path

import pytest

from topkd_imat.training.results import (
    COLUMNS,
    ResultsError,
    append_row,
    has_row,
    read_rows,
    run_name,
    uses_probe_head,
    validate_row,
    write_rows,
)

EXAMPLE = Path(__file__).with_name("results_example.csv")

ROW = {
    "dataset": "CIFAR-10",
    "condition": "KD",
    "supervision": "label-free",
    "init": "pretrained",
    "seed": 0,
    "accuracy": 0.9,
    "topology_distance": float("nan"),
}


def test_results_table_has_the_agreed_columns():
    assert COLUMNS == (
        "dataset",
        "condition",
        "supervision",
        "init",
        "seed",
        "accuracy",
        "topology_distance",
    )
    rows = read_rows(EXAMPLE)
    assert rows, "the example table must not be empty"


def test_example_table_covers_every_condition():
    rows = read_rows(EXAMPLE)
    assert {r["condition"] for r in rows} == {"Baseline", "KD", "Top", "KD+Top"}
    assert any(math.isnan(r["topology_distance"]) for r in rows), "NaN = not computed yet"


def test_probe_head_only_for_top_without_labels():
    assert uses_probe_head("Top", "label-free")
    assert not uses_probe_head("KD", "label-free")
    assert not uses_probe_head("KD+Top", "label-free")
    assert not uses_probe_head("Top", "label-supervised")


def test_append_writes_header_once_and_rows_round_trip(tmp_path):
    path = tmp_path / "results.csv"
    append_row(path, ROW)
    append_row(path, {**ROW, "seed": 1})
    rows = read_rows(path)
    assert [r["seed"] for r in rows] == [0, 1]
    assert path.read_text().count("dataset,condition") == 1
    assert has_row(path, ROW) and not has_row(path, {**ROW, "seed": 2})


def test_write_rows_replaces_the_table(tmp_path):
    path = tmp_path / "results.csv"
    append_row(path, ROW)
    write_rows(path, [{**r, "topology_distance": 0.5} for r in read_rows(path)])
    assert read_rows(path)[0]["topology_distance"] == 0.5


def test_run_name_is_unique_per_key():
    assert run_name(ROW) == "CIFAR-10_KD_label-free_pretrained_s0"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"condition": "TopKD"}, "condition must be one of"),
        ({"condition": "Baseline"}, "cannot be label-free"),
        ({"accuracy": 91.0}, r"fraction in \[0, 1\]"),
        ({"topology_distance": -1.0}, ">= 0 or NaN"),
        ({"seed": "abc"}, "seed must be an int"),
        ({"extra": 1}, "unexpected"),
    ],
)
def test_bad_rows_fail_with_a_readable_message(change, message):
    with pytest.raises(ResultsError, match=message):
        validate_row({**ROW, **change})
