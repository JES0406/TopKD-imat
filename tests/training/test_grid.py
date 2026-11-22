import math

import numpy as np

from topkd_imat.training.fill_distance import fill
from topkd_imat.training.grid import main, plan_runs
from topkd_imat.training.results import append_row, read_rows, run_name


def test_grid_has_seven_runs_per_init_and_no_label_free_baseline():
    runs = plan_runs(inits=("pretrained", "scratch"))
    assert len(runs) == 14
    assert not any(r["condition"] == "Baseline" and r["supervision"] == "label-free" for r in runs)
    assert len({run_name(r) for r in runs}) == 14


def test_skip_top_keeps_only_baseline_and_kd():
    runs = plan_runs(skip_top=True)
    assert {(r["condition"], r["supervision"]) for r in runs} == {
        ("Baseline", "label-supervised"),
        ("KD", "label-supervised"),
        ("KD", "label-free"),
    }


def test_grid_skips_runs_already_in_the_table(tmp_path, capsys):
    done = plan_runs(skip_top=True)[0]
    append_row(tmp_path / "results.csv", {**done, "accuracy": 0.5, "topology_distance": 0.1})
    assert main(["--skip-top", "--runs-dir", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 already done, 2 to go" in out
    assert f"[done] {run_name(done)}" in out


def _fake_runs_dir(tmp_path, n_pairs=2):
    rows = plan_runs(skip_top=True)
    for r in rows:
        append_row(tmp_path / "results.csv", {**r, "accuracy": 0.5, "topology_distance": math.nan})
        (tmp_path / run_name(r)).mkdir()
        clouds = {f"pair_{i}": np.ones((6, 3), np.float16) for i in range(n_pairs)}
        np.savez(tmp_path / run_name(r) / "clouds_test.npz", **clouds)
    (tmp_path / "teacher").mkdir()
    np.savez(
        tmp_path / "teacher" / "clouds_test.npz", pair_0=np.zeros((6, 5)), pair_1=np.zeros((6, 4))
    )
    return rows


def test_fill_distance_uses_the_last_layer_pair(tmp_path):
    _fake_runs_dir(tmp_path)
    seen = []

    def fake_distance(t, s):
        seen.append((t.shape, s.shape))
        return 0.25

    assert fill(tmp_path, fake_distance) == 3
    assert all(r["topology_distance"] == 0.25 for r in read_rows(tmp_path / "results.csv"))
    assert seen[0] == ((6, 4), (6, 3))


def test_fill_distance_with_the_real_topology_distance_fills_every_row(tmp_path):
    _fake_runs_dir(tmp_path)
    assert fill(tmp_path) == 3
    distances = [r["topology_distance"] for r in read_rows(tmp_path / "results.csv")]
    assert all(math.isfinite(d) and d >= 0 for d in distances)
