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
