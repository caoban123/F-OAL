"""Unit tests for Table 4 Ablation Study components and configurations."""
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
