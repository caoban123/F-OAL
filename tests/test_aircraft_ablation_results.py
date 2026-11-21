"""Checks on the saved FGVC-Aircraft results (Table 4, C1-C3, author code).

Reads only json/csv files under results/. No GPU, network, pretrained weights,
feature caches or W&B runs are needed, so it runs on a clean checkout.
"""
import csv
import json
import math
from pathlib import Path

import pytest

from src.foal.metrics import summary_from_task_matrix

ROOT = Path(__file__).resolve().parents[1]
TABLE4_DIR = ROOT / "results" / "aircraft_ablation"
SEEDS = (42, 123, 2026)
CONDITIONS = ("a0_full", "a1_without_ff", "a2_without_sp", "a3_without_both")
TABLE4_RUNS = [f"aircraft_{c}_seed{s}" for s in SEEDS for c in CONDITIONS]
SUPPLEMENTARY_RUNS = ["aircraft_a0_full_bilinear_seed42"]
ALIGNMENT_RUNS = [
    "aircraft_c1_full/p0_gaussian_std1",
    "aircraft_c1_full/p1_official_uniform",
    "aircraft_c1_full/p2_gaussian_matched_variance",
    "aircraft_c2_full/b1_augreg2_in21k_ft_in1k",
    "aircraft_c3_full/c3_b_bilinear_normalized",
    "aircraft_c3_full/c3_c_bicubic_raw",
]
OFFICIAL_DIR = ROOT / "results" / "official_foal_aircraft_seed0"
NUM_TRAIN, NUM_TEST = 3334, 3333
TOL = 1e-9


def _run_dirs():
    dirs = [TABLE4_DIR / r for r in TABLE4_RUNS + SUPPLEMENTARY_RUNS]
    dirs += [ROOT / "results" / r for r in ALIGNMENT_RUNS]
    return dirs


def _load(run_dir: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return metrics, metadata


def _matrix_from_csv(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            rows.setdefault(int(rec["stage"]), {})[int(rec["task"])] = float(rec["accuracy"])
    return [[rows[s][t] for t in range(1, s + 1)] for s in sorted(rows)]


def test_all_table4_runs_present():
    """A0-A3 exist for seeds 42, 123 and 2026, and the bilinear run is kept apart."""
    for name in TABLE4_RUNS + SUPPLEMENTARY_RUNS:
        for f in ("metrics.json", "metadata.json", "task_accuracy.csv"):
            assert (TABLE4_DIR / name / f).is_file(), f"missing {name}/{f}"
    assert len(TABLE4_RUNS) == 12


@pytest.mark.parametrize("run_dir", _run_dirs(), ids=lambda p: p.name)
def test_matrix_shape_range_and_recomputed_metrics(run_dir):
    """10 stages, 55 values in [0, 1], and every summary metric recomputes from the matrix."""
    metrics, _ = _load(run_dir)
    matrix = metrics["task_matrix"]
    assert len(matrix) == 10
    assert sum(len(r) for r in matrix) == 55
    assert all(0.0 <= v <= 1.0 for r in matrix for v in r)
    recomputed = summary_from_task_matrix(matrix)
    for key in ("average_incremental_accuracy", "last_accuracy", "forgetting_past_tasks", "forgetting_official_code"):
        assert math.isclose(metrics[key], recomputed[key], abs_tol=TOL), key


@pytest.mark.parametrize("run_dir", _run_dirs(), ids=lambda p: p.name)
def test_csv_matches_json(run_dir):
    """task_accuracy.csv holds the same 55 numbers as metrics.json."""
    metrics, _ = _load(run_dir)
    from_csv = _matrix_from_csv(run_dir / "task_accuracy.csv")
    assert len(from_csv) == 10
    for a, b in zip(metrics["task_matrix"], from_csv):
        assert len(a) == len(b)
        assert all(math.isclose(x, y, abs_tol=TOL) for x, y in zip(a, b))


@pytest.mark.parametrize("run_dir", _run_dirs(), ids=lambda p: p.name)
def test_split_sizes_match_official_train_test(run_dir):
    """Per-task sample counts add up to the official train (3,334) and test (3,333) splits."""
    _, metadata = _load(run_dir)
    tasks = metadata["task_split_fingerprint"]["tasks"]
    assert len(tasks) == 10
    assert sum(t["num_train_samples"] for t in tasks) == NUM_TRAIN
    assert sum(t["num_test_samples"] for t in tasks) == NUM_TEST
    labels = [l for t in tasks for l in t["new_original_labels"]]
    assert sorted(labels) == list(range(100)), "each of the 100 classes appears in exactly one task"


@pytest.mark.parametrize("seed", SEEDS)
def test_same_split_and_class_order_within_seed(seed):
    """A0-A3 of one seed share the class order and the split fingerprint."""
    fps = set()
    for c in CONDITIONS:
        _, md = _load(TABLE4_DIR / f"aircraft_{c}_seed{seed}")
        fp = md["task_split_fingerprint"]
        fps.add((fp["sha256"], fp["class_order_sha256"]))
    assert len(fps) == 1


def test_alignment_runs_share_seed42_split():
    """C1-C3 and the bilinear run use the same split as the Table 4 seed-42 runs."""
    _, ref = _load(TABLE4_DIR / "aircraft_a0_full_seed42")
    want = ref["task_split_fingerprint"]["sha256"]
    for run_dir in [TABLE4_DIR / r for r in SUPPLEMENTARY_RUNS] + [ROOT / "results" / r for r in ALIGNMENT_RUNS]:
        _, md = _load(run_dir)
        assert md["task_split_fingerprint"]["sha256"] == want, run_dir.name


def test_summary_csv_matches_metrics():
    """Every row of results/aircraft_ablation/summary.csv equals its run's metrics.json."""
    with (TABLE4_DIR / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13
    for row in rows:
        metrics, _ = _load(TABLE4_DIR / f"aircraft_{row['name']}_seed{row['seed']}")
        for key in ("average_incremental_accuracy", "last_accuracy", "forgetting_past_tasks", "forgetting_official_code"):
            assert math.isclose(float(row[key]), metrics[key], abs_tol=TOL), (row["name"], row["seed"], key)


def test_table4_means_match_report():
    """Mean A_avg over 3 seeds, as reported in docs (64.22 / 62.98 / 60.40 / 57.75)."""
    expected = {"a0_full": 64.22, "a1_without_ff": 62.98, "a2_without_sp": 60.40, "a3_without_both": 57.75}
    for c, value in expected.items():
        accs = [_load(TABLE4_DIR / f"aircraft_{c}_seed{s}")[0]["average_incremental_accuracy"] * 100 for s in SEEDS]
        assert round(sum(accs) / 3, 2) == value


def test_official_run_matches_author_printout():
    """The author-code run: our recomputation equals the numbers general_main.py printed.

    The matrix was parsed from the printed numpy rows (8 decimals), hence the 1e-6 tolerance.
    """
    metrics = json.loads((OFFICIAL_DIR / "metrics.json").read_text(encoding="utf-8"))
    matrix = metrics["task_matrix"]
    assert sum(len(r) for r in matrix) == 55
    recomputed = summary_from_task_matrix(matrix)
    printed = metrics["author_printed"]
    assert math.isclose(recomputed["average_incremental_accuracy"], printed["avg_acc"], abs_tol=1e-6)
    assert math.isclose(recomputed["last_accuracy"], printed["avg_end_acc"], abs_tol=1e-6)
    assert math.isclose(recomputed["forgetting_official_code"], printed["avg_end_fgt"], abs_tol=1e-6)
