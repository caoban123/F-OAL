"""Unit and regression tests for CIFAR-100 Multi-Seed Table 4 Ablation pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analyze_cifar100_multiseed import (
    ABLATION_ORDER,
    compute_statistical_summary,
    generate_figures,
    load_all_runs,
    sample_mean_std,
    save_csv_outputs,
)
from scripts.run_cifar100_multiseed_ablation import (
    ABLATIONS,
    EXPECTED_BACKBONE,
    EXPECTED_INTERPOLATION,
    EXPECTED_NORMALIZE,
    EXPECTED_PROJECTION_DIM,
    EXPECTED_PROJECTION_INIT,
    build_seed_manifest,
    validate_completed_run,
)
from src.foal.metrics import summary_from_task_matrix


def _write_synthetic_run(
    run_dir: Path,
    *,
    seed: int,
    ablation_key: str,
    offset: float = 0.0,
    is_pilot: bool = False,
    max_tasks: int = 10,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    abl = ABLATIONS[ablation_key]

    # Deterministic lower-triangular 10x10 matrix
    matrix: list[list[float]] = []
    for s in range(max_tasks):
        row = []
        for t in range(s + 1):
            val = min(0.99, max(0.50, 0.92 - 0.006 * (s - t) + offset))
            row.append(round(val, 4))
        matrix.append(row)

    with (run_dir / "task_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "task", "accuracy"])
        for s_idx, row in enumerate(matrix, start=1):
            for t_idx, val in enumerate(row, start=1):
                w.writerow([s_idx, t_idx, val])

    metrics = summary_from_task_matrix(matrix)
    metrics["total_duration_seconds"] = 123.45
    (run_dir / "metrics.json").write_text(json.dumps({"task_matrix": matrix, **metrics}, indent=2), encoding="utf-8")

    meta = {
        "experiment_type": "pilot" if is_pilot else "ablation",
        "seeds": {"base": seed, "class_order": seed, "projection": seed, "stream": seed},
        "model_config": {
            "backbone": EXPECTED_BACKBONE,
            "feature_fusion": abl["feature_fusion"],
            "smooth_projection": abl["smooth_projection"],
            "projection_init": EXPECTED_PROJECTION_INIT if abl["smooth_projection"] else "gaussian_std1",
            "projection_dim": EXPECTED_PROJECTION_DIM if abl["smooth_projection"] else 768,
            "gamma": 1.0,
            "rls_dtype": "float64",
        },
        "projection_metadata": {"matrix_sha256": f"proj_{seed}_{ablation_key}"} if abl["smooth_projection"] else None,
        "training_protocol": {
            "max_tasks": max_tasks,
            "limit_train_per_task": 50 if is_pilot else None,
            "limit_test_per_task": 50 if is_pilot else None,
        },
        "task_split_fingerprint": {
            "sha256": f"split_fp_seed_{seed}",
            "class_order_sha256": f"class_order_seed_{seed}",
        },
        "config": {
            "dataset": {
                "interpolation": EXPECTED_INTERPOLATION,
                "normalize": EXPECTED_NORMALIZE,
            }
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for t in range(1, max_tasks + 1):
        torch.save({"task_number": t}, run_dir / f"task_{t:02d}.pt")


def test_seed42_manifest_and_audit(tmp_path: Path) -> None:
    manifest = build_seed_manifest(tmp_path)
    assert manifest["reused_seed"] == 42
    s42 = manifest["seed_42_verified_runs"]
    assert set(s42.keys()) == {"full", "no_ff", "no_sp", "no_ff_no_sp"}
    assert abs(s42["full"]["average_incremental_accuracy_pct"] - 90.401635) < 1e-4
    assert abs(s42["no_ff"]["average_incremental_accuracy_pct"] - 90.100369) < 1e-4
    assert abs(s42["no_sp"]["average_incremental_accuracy_pct"] - 90.382341) < 1e-4
    assert abs(s42["no_ff_no_sp"]["average_incremental_accuracy_pct"] - 89.922353) < 1e-4
    fps = {s42[k]["task_split_fingerprint"] for k in s42}
    assert len(fps) == 1
    assert (tmp_path / "seed_manifest.json").exists()


def test_validate_completed_run_guards(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed_123" / "full"
    _write_synthetic_run(run_dir, seed=123, ablation_key="full")
    ok, reasons, summary = validate_completed_run(run_dir, expected_seed=123, ablation_key="full")
    assert ok, reasons
    assert summary is not None

    # Wrong seed must fail
    ok_bad_seed, reasons_seed, _ = validate_completed_run(run_dir, expected_seed=2026, ablation_key="full")
    assert not ok_bad_seed
    assert any("Seed mismatch" in r for r in reasons_seed)

    # Wrong ablation must fail
    ok_bad_abl, reasons_abl, _ = validate_completed_run(run_dir, expected_seed=123, ablation_key="no_ff")
    assert not ok_bad_abl
    assert any("feature_fusion mismatch" in r for r in reasons_abl)

    # Missing checkpoint must fail
    (run_dir / "task_10.pt").unlink()
    ok_missing_ckpt, reasons_ckpt, _ = validate_completed_run(run_dir, expected_seed=123, ablation_key="full")
    assert not ok_missing_ckpt
    assert any("Missing checkpoint: task_10.pt" in r for r in reasons_ckpt)

    # Pilot contamination (e.g., 10 tasks but limited samples) must fail full validation
    pilot_dir = tmp_path / "pilot_run"
    _write_synthetic_run(pilot_dir, seed=123, ablation_key="full", is_pilot=True, max_tasks=10)
    ok_pilot, reasons_pilot, _ = validate_completed_run(pilot_dir, expected_seed=123, ablation_key="full", is_pilot=False)
    assert not ok_pilot
    assert any("Pilot contamination" in r for r in reasons_pilot)


def test_sample_mean_std_uses_ddof1() -> None:
    vals = [90.0, 91.0, 92.0]
    m, s = sample_mean_std(vals)
    assert m == pytest.approx(91.0)
    assert s == pytest.approx(1.0)


def test_end_to_end_analysis_and_figures(tmp_path: Path) -> None:
    offsets = {"full": 0.01, "no_ff": 0.005, "no_sp": 0.008, "no_ff_no_sp": 0.0}
    for seed, seed_shift in [(123, 0.002), (2026, -0.003)]:
        for abl_key in ABLATION_ORDER:
            _write_synthetic_run(
                tmp_path / f"seed_{seed}" / abl_key,
                seed=seed,
                ablation_key=abl_key,
                offset=offsets[abl_key] + seed_shift,
            )

    data = load_all_runs(tmp_path)
    stats = compute_statistical_summary(data)

    assert stats["std_ddof"] == 1
    assert stats["sample_size_n"] == 3
    pe = stats["paired_effects"]["average_incremental_accuracy_pct"]
    assert pe["delta_ff"]["positive_seeds"] == 3
    assert pe["delta_both"]["positive_seeds"] == 3

    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csvs = save_csv_outputs(stats, out_dir)
    figs = generate_figures(data, stats, out_dir)

    assert all(p.exists() and p.stat().st_size > 0 for p in csvs)
    assert len(figs) == 4
    assert all(p.exists() and p.stat().st_size > 1000 for p in figs)


def test_multiseed_notebook_integrity() -> None:
    nb_path = Path("notebooks/foal_cifar100_multiseed_ablation.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    full_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for expected_token in [
        "Preflight",
        "Pilot",
        "Full Experiments",
        "Validation",
        "foal_cifar100_multiseed_ablation_results.zip",
        "scripts.run_cifar100_multiseed_ablation",
        "scripts.analyze_cifar100_multiseed",
    ]:
        assert expected_token in full_src
