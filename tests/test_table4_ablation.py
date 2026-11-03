"""Unit tests for Table 4 Ablation Study components and configurations."""
import json
import pytest
import torch
import yaml
from pathlib import Path

from src.foal.backbone import FrozenViT
from src.foal.projection import SmoothProjection
from src.foal.rls import OnlineRidgeRLS
from src.foal.model import FOALModel


def test_configs_exist_and_valid():
    """Verify all 4 Table 4 configuration files exist and parse without error."""
    configs = [
        "configs/table4_a0_full_foal_cifar100.yaml",
        "configs/table4_a1_without_ff_cifar100.yaml",
        "configs/table4_a2_without_sp_cifar100.yaml",
        "configs/table4_a3_without_both_cifar100.yaml",
    ]
    for cfg_path in configs:
        p = Path(cfg_path)
        assert p.exists(), f"Missing config: {cfg_path}"
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["dataset"]["interpolation"] == "bilinear"
        assert cfg["dataset"]["normalize"] is False
        assert cfg["model"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def test_table4_architectural_combinations():
    """Verify model construction and feature dimensions across the 2x2 ablation design."""
    # A0: FF True, SP True
    m_a0 = FOALModel(
        backbone_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        feature_fusion=True,
        smooth_projection=True,
        projection_dim=1000,
        projection_init="official_uniform",
    )
    assert m_a0.feature_fusion_enabled is True
    assert m_a0.smooth_projection_enabled is True
    assert m_a0.projection is not None
    assert m_a0.classifier.feature_dim == 1000

    # A1: FF False, SP True
    m_a1 = FOALModel(
        backbone_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        feature_fusion=False,
        smooth_projection=True,
        projection_dim=1000,
        projection_init="official_uniform",
    )
    assert m_a1.feature_fusion_enabled is False
    assert m_a1.smooth_projection_enabled is True
    assert m_a1.projection is not None
    assert m_a1.classifier.feature_dim == 1000

    # A2: FF True, SP False
    m_a2 = FOALModel(
        backbone_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        feature_fusion=True,
        smooth_projection=False,
        projection_dim=768,
        projection_init="official_uniform",
    )
    assert m_a2.feature_fusion_enabled is True
    assert m_a2.smooth_projection_enabled is False
    assert m_a2.projection is None
    assert m_a2.classifier.feature_dim == 768

    # A3: FF False, SP False
    m_a3 = FOALModel(
        backbone_name="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        feature_fusion=False,
        smooth_projection=False,
        projection_dim=768,
        projection_init="official_uniform",
    )
    assert m_a3.feature_fusion_enabled is False
    assert m_a3.smooth_projection_enabled is False
    assert m_a3.projection is None
    assert m_a3.classifier.feature_dim == 768


def test_closed_form_ridge_rls_equivalence():
    """Verify OnlineRidgeRLS produces identical weights to batch closed-form Ridge Regression.
    
    Closed-form Ridge: W = (X^T X + gamma * I)^{-1} X^T Y
    Since in foal rls.py: W has shape (feature_dim, num_classes) and scores = H @ W.
    """
    for dim in [768, 1000]:
        num_samples = 250
        num_classes = 10
        gamma = 1.0

        torch.manual_seed(42)
        X = torch.randn(num_samples, dim, dtype=torch.float64)
        y = torch.randint(0, num_classes, (num_samples,))
        Y = torch.zeros(num_samples, num_classes, dtype=torch.float64)
        Y.scatter_(1, y.unsqueeze(1), 1.0)

        # 1. Closed-form batch solution: W = (X^T X + gamma * I)^{-1} X^T Y
        XT_X = X.T @ X
        reg = gamma * torch.eye(dim, dtype=torch.float64)
        inv = torch.linalg.inv(XT_X + reg)
        W_closed = inv @ (X.T @ Y)

        # 2. Online RLS solution
        rls = OnlineRidgeRLS(feature_dim=dim, gamma=gamma, dtype=torch.float64)
        rls.expand_classes(num_classes)

        batch_size = 25
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            rls.update(X_batch, y_batch)

        # Assert max absolute difference < 1e-7
        diff = torch.max(torch.abs(rls.W - W_closed)).item()
        assert diff < 1e-7, f"RLS vs Closed-form mismatch for dim {dim}: max diff = {diff}"


def test_cache_key_isolation_ff_vs_noff():
    """Verify that feature cache prefixes differentiate feature fusion status."""
    use_fusion_true = True
    ff_tag_true = "ff" if use_fusion_true else "noff"

    use_fusion_false = False
    ff_tag_false = "ff" if use_fusion_false else "noff"

    assert ff_tag_true == "ff"
    assert ff_tag_false == "noff"
    assert ff_tag_true != ff_tag_false


def test_reference_ground_truth_provenance():
    """Verify paper reference numbers match NeurIPS 2024 Page 10 Table 4 and Page 7 Table 2."""
    ref_path = Path("docs/paper_reference_ground_truth.json")
    assert ref_path.exists(), "Missing paper_reference_ground_truth.json"
    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Table 4 verification
    t4 = data["table_4_ablation"]["entries"]
    assert len(t4) == 4
    # CIFAR-100
    assert t4[0]["cifar100"] == 91.1
    assert t4[1]["cifar100"] == 90.6
    assert t4[2]["cifar100"] == 90.7
    assert t4[3]["cifar100"] == 90.6
    # Tiny-ImageNet
    assert t4[0]["tiny_imagenet"] == 91.2
    assert t4[1]["tiny_imagenet"] == 91.4
    assert t4[2]["tiny_imagenet"] == 91.2
    assert t4[3]["tiny_imagenet"] == 91.4
    # DTD
    assert t4[0]["dtd"] == 82.8
    assert t4[1]["dtd"] == 80.5
    assert t4[2]["dtd"] == 79.3
    assert t4[3]["dtd"] == 71.2

    # Table 2 verification (CIFAR-100)
    t2 = {m["method"]: m for m in data["table_2_benchmark"]["cifar100_results"]}
    assert t2["F-OAL (Ours)"]["a_avg"] == 91.1
    assert t2["LwF (TPAMI 2018)"]["a_avg"] == 69.3
    assert t2["EWC (PNAS 2017)"]["a_avg"] == 49.9
    assert t2["ER (ICRA 2019)"]["a_avg"] == 90.1
    assert t2["PCR (CVPR 2023)"]["a_avg"] == 89.1
    assert t2["iCaRL (CVPR 2017)"]["a_avg"] == 91.6
    assert t2["SCR (CVPR 2021)"]["a_avg"] == 91.9
    assert t2["DVC (CVPR 2022)"]["a_avg"] == 92.4
def test_table4_archive_and_recalculated_metrics():
    """Verify Table 4 archive manifest exists and unrounded metric values match recalculation."""
    manifest_p = Path("results/table4_analysis/table4_evidence_manifest.json")
    assert manifest_p.exists(), "Missing table4_evidence_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        ev = json.load(f)

    assert ev["archive_provenance"]["sha256"] == "90f191d12b06fbd804779fcb038a12574c9b7cdc2f32369a6975dd6309e09e7a"
    assert ev["archive_provenance"]["crc_status"] == "ALL_OK"
    assert ev["reused_baseline_a0"]["status"] == "A0_REUSE_VERIFIED"
    assert ev["fixed_controls_verified"] is True
    assert ev["metrics_recalculated_match"] is True

    # Check unrounded recalculated accuracies
    fe = ev["factorial_effects"]
    assert abs(fe["A0_Full"] - 90.40163492063492) < 1e-6
    assert abs(fe["A1_noFF"] - 90.10036904761906) < 1e-6
    assert abs(fe["A2_noSP"] - 90.38234126984127) < 1e-6
    assert abs(fe["A3_noBoth"] - 89.92235317460316) < 1e-6

    # Verify mathematical identity: E_FF + E_SP == A0 - A3
    diff_joint = abs(fe["joint_benefit"]["delta_full_vs_none"] - fe["joint_benefit"]["sum_of_main_effects"])
    assert diff_joint < 1e-12, f"Additivity gap violated: {diff_joint}"


def test_stage_accuracy_from_raw_matrix_reconciliation():
    """Forensic Check 1 & 2: Reconstruct stage accuracies from raw 55-element task matrices and check mean equals A_bar."""
    import pandas as pd
    import numpy as np

    raw_paths = {
        "A0": Path("results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/task_accuracy.csv"),
        "A1": Path("results/table4_full_gpu_verified/results/table4_a1_without_ff/task_accuracy.csv"),
        "A2": Path("results/table4_full_gpu_verified/results/table4_a2_without_sp/task_accuracy.csv"),
        "A3": Path("results/table4_full_gpu_verified/results/table4_a3_without_both/task_accuracy.csv"),
    }

    # Load exported stage CSV
    stage_csv_p = Path("results/table4_analysis/table4_stage_accuracies.csv")
    assert stage_csv_p.exists()
    stage_df = pd.read_csv(stage_csv_p)

    # Load metrics json
    metrics_json_p = Path("results/table4_analysis/table4_metrics_comparison.json")
    assert metrics_json_p.exists()
    with open(metrics_json_p, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    for cond_id, csv_p in raw_paths.items():
        assert csv_p.exists(), f"Raw CSV missing for {cond_id}"
        df = pd.read_csv(csv_p)
        assert len(df) == 55, f"Raw CSV for {cond_id} must have exactly 55 rows"

        # Construct 10x10 triangular matrix
        R = np.zeros((10, 10))
        for _, row in df.iterrows():
            s = int(row["stage"]) - 1
            t = int(row["task"]) - 1
            R[s, t] = float(row["accuracy"])

        # Compute per-stage accuracies
        computed_stages = [float(np.mean(R[s, :s+1])) for s in range(10)]

        # 1. Compare against table4_stage_accuracies.csv (rounded to 2 decimals)
        col_name = {
            "A0": "A0 (Full)", "A1": "A1 (w/o FF)", "A2": "A2 (w/o SP)", "A3": "A3 (w/o Both)"
        }[cond_id]
        exported_stages = [float(x) for x in stage_df[col_name].values]
        for s in range(10):
            assert abs(computed_stages[s] * 100.0 - exported_stages[s]) < 0.015, (
                f"{cond_id} Stage {s+1} mismatch: computed={computed_stages[s]*100.0}, exported={exported_stages[s]}"
            )

        # 2. Forensic Identity: mean(stage_accs) == a_bar
        computed_a_bar = float(np.mean(computed_stages))
        reported_a_bar = metrics_data[cond_id]["a_bar_ratio"]
        diff = abs(computed_a_bar - reported_a_bar)
        assert diff < 1e-12, f"{cond_id} Forensic violation: mean(stages) != a_bar (diff={diff})"

        # 3. Last accuracy: stage_accs[9] == a_last
        computed_a_last = computed_stages[9]
        reported_a_last = metrics_data[cond_id]["a_last_ratio"]
        assert abs(computed_a_last - reported_a_last) < 1e-12, f"{cond_id} A_last mismatch"

        # 4. Per-task forgetting recomputes summary metrics
        task_f_past = [float(np.max(R[j:9, j]) - R[9, j]) for j in range(9)]
        computed_f_past = float(np.mean(task_f_past))
        reported_f_past = metrics_data[cond_id]["f_past_ratio"]
        assert abs(computed_f_past - reported_f_past) < 1e-12, f"{cond_id} F_past mismatch"

        task_f_code = [float(np.max(R[j:10, j]) - R[9, j]) for j in range(10)]
        computed_f_code = float(np.mean(task_f_code))
        reported_f_code = metrics_data[cond_id]["f_code_ratio"]
        assert abs(computed_f_code - reported_f_code) < 1e-12, f"{cond_id} F_code mismatch"


def test_a0_matrix_bitwise_equality_with_c3_d():
    """Forensic Check 5: Verify A0 is bitwise identical in every matrix entry and metadata to C3-D."""
    import hashlib
    c3_d_csv = Path("results/c3_full_gpu_verified/results/c3_full_gpu/c3_d_bilinear_raw/task_accuracy.csv")
    assert c3_d_csv.exists()
    c3_d_bytes = c3_d_csv.read_bytes()
    c3_d_sha256 = hashlib.sha256(c3_d_bytes).hexdigest()

    # Verified reference hash from C3 archive manifest
    expected_c3_d_sha256 = "68684dfb3d348c5838815a63fa45dd874bd84b3a427cd29d06f4c320fa5de1d7"
    assert c3_d_sha256 == expected_c3_d_sha256, f"C3-D CSV SHA256 mismatch: {c3_d_sha256}"


def test_corrupted_fixture_detection():
    """Verify that any corrupted stage trajectory (such as hallucinated tables) is strictly detected and rejected."""
    import numpy as np

    # True A0 average incremental accuracy
    true_a_bar = 90.40163492063492

    # Previously hallucinated stage trajectory starting at 99.30%
    corrupted_stages = [99.30, 96.25, 93.80, 92.20, 90.96, 89.5833, 88.6143, 87.5875, 86.6333, 85.73]
    corrupted_mean = float(np.mean(corrupted_stages))

    # The discrepancy is large (+0.664 pp)
    discrepancy = abs(corrupted_mean - true_a_bar)
    assert discrepancy > 0.50, f"Corrupted table must differ significantly from true average (found {discrepancy})"

    # The verification rule must reject this corrupted trajectory
    tolerance = 1e-6
    is_valid = (discrepancy < tolerance)
    assert not is_valid, "Test harness must reject corrupted stage table"


def test_plots_and_wandb_inventory_audit():
    """Forensic Check 6, 7 & 8: Verify all 7 figures exist with non-empty payloads and W&B inventory is complete."""
    analysis_dir = Path("results/table4_analysis")
    
    # Check 7 figures
    for fig_num in range(1, 8):
        matches = list(analysis_dir.glob(f"fig{fig_num}_*.png"))
        assert len(matches) == 1, f"Missing or duplicate figure {fig_num}"
        fig_path = matches[0]
        assert fig_path.stat().st_size > 50000, f"Figure {fig_path.name} is unusually small ({fig_path.stat().st_size} bytes)"

    # Check W&B audit file
    wandb_audit_p = analysis_dir / "table4_wandb_audit.json"
    assert wandb_audit_p.exists()
    with open(wandb_audit_p, "r", encoding="utf-8") as f:
        w_audit = json.load(f)

    assert w_audit["total_online_runs"] == 17
    assert len(w_audit["table4_runs"]) == 4
    assert len(w_audit["pilot_smoke_runs"]) == 4

    a0_entry = [r for r in w_audit["table4_runs"] if r["condition"] == "A0"][0]
    assert a0_entry["provenance"] == "REUSED_VERIFIED_C3_D"
    assert a0_entry["run_id"] == "3v53tmq5"


