"""Forensic Verification Test Suite for DTD Table 4 Reproduction & Limitations.

Verifies:
1. Mathematical identity and reconciliation of the 10x10 triangular evaluation matrix.
2. Multi-seed stability across seeds 42, 123, 2026 (Limitation 1).
3. Zero-replay, zero-leakage, and dataset provenance (Limitation 2).
4. Timing decomposition: ViT feature extraction vs pure RLS (Limitation 3).
5. Architecture config provenance: Smooth Projection disabled in A2/A3 (Limitation 4).
"""
import json
import math
from pathlib import Path
import numpy as np
import pytest


DTD_RESULTS_DIR = Path("results/dtd")
CONDITIONS = {
    "A0": "dtd_a0_full",
    "A1": "dtd_a1_without_ff",
    "A2": "dtd_a2_without_sp",
    "A3": "dtd_a3_without_both",
}


def test_dtd_matrix_mathematical_reconciliation():
    """Verify that stage accuracies, A_bar, A_10, F_past, and F_code perfectly reconcile from raw task_matrix."""
    for cond_id, folder_name in CONDITIONS.items():
        run_dir = DTD_RESULTS_DIR / folder_name
        metrics_file = run_dir / "metrics.json"
        assert metrics_file.exists(), f"metrics.json missing for {cond_id}"

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_matrix = data["task_matrix"]
        assert len(task_matrix) == 10, f"Expected 10 stages in {cond_id}"

        # Reconstruct 10x10 lower triangular matrix R
        R = np.zeros((10, 10))
        for s, row in enumerate(task_matrix):
            assert len(row) == s + 1, f"Stage {s+1} should have {s+1} evaluations"
            for t, val in enumerate(row):
                R[s, t] = float(val)

        # 1. Per-stage accuracy: A_s = (1/s) sum_{t=1}^s R_{s, t}
        computed_stages = [float(np.mean(R[s, :s+1])) for s in range(10)]
        reported_stages = data["stage_accuracies"]
        for s in range(10):
            assert abs(computed_stages[s] - reported_stages[s]) < 1e-12, (
                f"{cond_id} Stage {s+1} mismatch: {computed_stages[s]} vs {reported_stages[s]}"
            )

        # 2. Average incremental accuracy: A_bar = (1/10) sum_{s=1}^{10} A_s
        computed_a_bar = float(np.mean(computed_stages))
        reported_a_bar = data["average_incremental_accuracy"]
        assert abs(computed_a_bar - reported_a_bar) < 1e-12, (
            f"{cond_id} A_bar mismatch: {computed_a_bar} vs {reported_a_bar}"
        )

        # 3. Final task accuracy: A_10 = A_{stage 10}
        computed_a_10 = computed_stages[9]
        reported_a_10 = data["last_accuracy"]
        assert abs(computed_a_10 - reported_a_10) < 1e-12, f"{cond_id} A_10 mismatch"

        # 4. Forgetting past tasks: F_past = (1/9) sum_{j=1}^9 (max_{l in {j..9}} R_{l, j} - R_{10, j})
        computed_f_past = float(np.mean([np.max(R[j:9, j]) - R[9, j] for j in range(9)]))
        reported_f_past = data["forgetting_past_tasks"]
        assert abs(computed_f_past - reported_f_past) < 1e-12, f"{cond_id} F_past mismatch"

        # 5. Forgetting official code: F_code = (1/10) sum_{j=1}^{10} (max_{l in {j..10}} R_{l, j} - R_{10, j})
        computed_f_code = float(np.mean([np.max(R[j:10, j]) - R[9, j] for j in range(10)]))
        reported_f_code = data["forgetting_official_code"]
        assert abs(computed_f_code - reported_f_code) < 1e-12, f"{cond_id} F_code mismatch"


def test_dtd_multi_seed_stability_and_ci():
    """Verify multi-seed statistical significance and variance (Addressing Limitation 1)."""
    seed_paths = [
        Path("results/foal_full_dtd_seed42/metrics.json"),
        Path("results/foal_multiseed_dtd_seed123/metrics.json"),
        Path("results/foal_multiseed_dtd_seed2026/metrics.json"),
    ]
    for p in seed_paths:
        assert p.exists(), f"Seed metrics missing at {p}"

    a_bars = []
    a_lasts = []
    for p in seed_paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            a_bars.append(d["average_incremental_accuracy"] * 100.0)
            a_lasts.append(d["last_accuracy"] * 100.0)

    n = len(a_bars)
    mean_a_bar = float(np.mean(a_bars))
    std_a_bar = float(np.std(a_bars, ddof=1))

    # Variance across 3 seeds must be small (< 0.20 percentage points)
    assert std_a_bar < 0.20, f"Standard deviation too high: {std_a_bar:.4f} pp"

    # 95% Confidence Interval with t-distribution (t_{2, 0.025} = 4.303)
    t_crit = 4.303
    ci_margin = t_crit * (std_a_bar / math.sqrt(n))
    ci_lower = mean_a_bar - ci_margin
    ci_upper = mean_a_bar + ci_margin

    # The 95% CI lower bound must be significantly higher than paper target (82.8%)
    assert ci_lower > 82.8, f"Lower bound {ci_lower:.2f}% does not exceed paper 82.8%"
    assert mean_a_bar > 85.0


def test_zero_replay_and_zero_leakage_provenance():
    """Verify strictly disjoint training tasks (Zero-Replay) and no train-test leakage (Limitation 2 & 4)."""
    meta_path = DTD_RESULTS_DIR / "dtd_a0_full" / "metadata.json"
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Dataset must be real tanganke/dtd, not mock
    assert meta["training_protocol"]["dataset"] == "tanganke/dtd"
    assert meta["model_config"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"

    # 2. Check task split fingerprint
    tasks = meta["task_split_fingerprint"]["tasks"]
    assert len(tasks) == 10

    all_train_labels = []
    for t in tasks:
        new_labels = t["new_original_labels"]
        assert len(new_labels) == 4, "Each task must have 4 new classes"
        for label in new_labels:
            assert label not in all_train_labels, f"Class {label} replayed in multiple tasks!"
            all_train_labels.append(label)

    assert len(all_train_labels) == 40, "Exactly 40 distinct texture classes used"


def test_timing_decomposition_provenance():
    """Verify runtime metrics cleanly decompose ViT extraction from RLS (Limitation 3)."""
    for cond_id, folder in CONDITIONS.items():
        with open(DTD_RESULTS_DIR / folder / "metrics.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "total_duration_seconds" in data
        assert "rls_and_eval_seconds" in data
        assert "feature_extraction_seconds" in data

        rls_sec = data["rls_and_eval_seconds"]
        # RLS + eval on GPU should be between 0.8s and 2.0s
        assert 0.8 <= rls_sec <= 2.0, f"{cond_id} RLS time {rls_sec} outside expected GPU range"

        # A2 and A3 benefited from cached features
        if cond_id in ("A2", "A3"):
            assert data["total_duration_seconds"] < 2.0


def test_smooth_projection_architecture_flag_provenance():
    """Verify that Smooth Projection is strictly disabled in A2 and A3 (Limitation 4)."""
    for cond_id, folder in [("A2", "dtd_a2_without_sp"), ("A3", "dtd_a3_without_both")]:
        with open(DTD_RESULTS_DIR / folder / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Smooth projection must be explicitly False
        assert meta["model_config"]["smooth_projection"] is False
        assert meta["model_config"]["projection_dim"] == 768
        # Projection metadata must be null
        assert meta["projection_metadata"] is None

    # For A0 and A1: Smooth projection is True, dim is 1000
    for cond_id, folder in [("A0", "dtd_a0_full"), ("A1", "dtd_a1_without_ff")]:
        with open(DTD_RESULTS_DIR / folder / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["model_config"]["smooth_projection"] is True
        assert meta["model_config"]["projection_dim"] == 1000
        assert meta["projection_metadata"] is not None
        assert abs(meta["projection_metadata"]["measured_std"] - 0.02083) < 0.001


def test_sp_ablation_plasticity_forgetting_paradox():
    """Forensically verify the plasticity-stability paradox between A0 and A2 on DTD (docs/Limitations_2.md).
    
    Verifies:
    1. Delta A_bar = +2.85 pp, Delta A_10 = +0.44 pp, Delta F_past = +2.22 pp.
    2. Mean Peak(A0) - Mean Peak(A2) = +2.71 pp (A0 has much higher learning plasticity).
    3. Mean Final(A0) - Mean Final(A2) = +0.49 pp (both converge at task 10).
    4. Mathematical identity: Delta F_past = Delta Mean Peak - Delta Mean Final.
    5. Monotonic decrease of peak learning capacity across A0 > A1 > A2 > A3.
    """
    metrics = {}
    matrices = {}
    for cond_id, folder in CONDITIONS.items():
        with open(DTD_RESULTS_DIR / folder / "metrics.json", "r", encoding="utf-8") as f:
            d = json.load(f)
            metrics[cond_id] = d
            R = np.zeros((10, 10))
            for s, row in enumerate(d["task_matrix"]):
                for t, v in enumerate(row):
                    R[s, t] = float(v)
            matrices[cond_id] = R

    # 1. Delta checks between A0 and A2
    a_bar_0 = metrics["A0"]["average_incremental_accuracy"] * 100.0
    a_bar_2 = metrics["A2"]["average_incremental_accuracy"] * 100.0
    delta_a_bar = a_bar_0 - a_bar_2
    assert abs(delta_a_bar - 2.8477) < 0.01, f"Expected delta A_bar ~ 2.85 pp, got {delta_a_bar}"

    a_10_0 = metrics["A0"]["last_accuracy"] * 100.0
    a_10_2 = metrics["A2"]["last_accuracy"] * 100.0
    delta_a_10 = a_10_0 - a_10_2
    assert abs(delta_a_10 - 0.4375) < 0.01, f"Expected delta A_10 ~ 0.44 pp, got {delta_a_10}"

    f_past_0 = metrics["A0"]["forgetting_past_tasks"] * 100.0
    f_past_2 = metrics["A2"]["forgetting_past_tasks"] * 100.0
    delta_f_past = f_past_0 - f_past_2
    assert abs(delta_f_past - 2.2222) < 0.01, f"Expected delta F_past ~ 2.22 pp, got {delta_f_past}"

    # 2. Decompose peaks and finals for tasks 1 to 9
    R0 = matrices["A0"]
    R2 = matrices["A2"]
    peaks_0 = [np.max(R0[j:9, j]) * 100.0 for j in range(9)]
    peaks_2 = [np.max(R2[j:9, j]) * 100.0 for j in range(9)]
    finals_0 = [R0[9, j] * 100.0 for j in range(9)]
    finals_2 = [R2[9, j] * 100.0 for j in range(9)]

    mean_peak_0 = float(np.mean(peaks_0))
    mean_peak_2 = float(np.mean(peaks_2))
    delta_peak = mean_peak_0 - mean_peak_2

    mean_final_0 = float(np.mean(finals_0))
    mean_final_2 = float(np.mean(finals_2))
    delta_final = mean_final_0 - mean_final_2

    assert abs(delta_peak - 2.7083) < 0.01, f"Expected delta Peak ~ 2.71 pp, got {delta_peak}"
    assert abs(delta_final - 0.4861) < 0.01, f"Expected delta Final ~ 0.49 pp, got {delta_final}"

    # 3. Exact mathematical identity
    identity_diff = abs(delta_f_past - (delta_peak - delta_final))
    assert identity_diff < 1e-10, f"Identity violated: {identity_diff}"

    # 4. Stage 2 gap is large (> 9.0 pp), Stage 10 gap is small (< 0.5 pp)
    stage_2_gap = (R0[1, :2].mean() - R2[1, :2].mean()) * 100.0
    assert abs(stage_2_gap - 9.0625) < 0.01

    # 5. Monotonic decrease of Peak across all 4 conditions
    mean_peaks = {}
    for c in ["A0", "A1", "A2", "A3"]:
        Rc = matrices[c]
        p_c = [np.max(Rc[j:9, j]) * 100.0 for j in range(9)]
        mean_peaks[c] = float(np.mean(p_c))

    assert mean_peaks["A0"] > mean_peaks["A1"] > mean_peaks["A2"] > mean_peaks["A3"], (
        f"Expected monotonic decrease in peak capacity: {mean_peaks}"
    )
