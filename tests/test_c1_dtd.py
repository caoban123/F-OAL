"""Unit and regression tests for Experiment C1 on DTD (Smooth Projection Init)."""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import yaml

from src.foal.projection import SmoothProjection, VALID_PROJECTION_INITIALIZATIONS
from src.foal.diagnostics import compare_c1_projections_on_features


CONFIGS_DIR = Path("configs/dtd")


def test_c1_dtd_configs_exist_and_parse():
    """Verify all three C1 DTD configs exist and parse valid YAML."""
    c1_files = [
        CONFIGS_DIR / "c1_p0_gaussian_std1.yaml",
        CONFIGS_DIR / "c1_p1_official_uniform.yaml",
        CONFIGS_DIR / "c1_p2_gaussian_matched_variance.yaml",
    ]
    for p in c1_files:
        assert p.exists(), f"Missing config: {p}"
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["dataset"]["type"] == "dtd"
        assert data["dataset"]["name"] == "tanganke/dtd"
        assert data["dataset"]["samples_per_class"] == 40
        assert data["dataset"]["classes_per_task"] == 4
        assert data["dataset"]["num_classes"] == 40
        assert data["model"]["backbone"] == "vit_base_patch16_224.augreg_in1k"
        assert data["model"]["feature_fusion"] is True
        assert data["model"]["smooth_projection"] is True
        assert data["model"]["projection_dim"] == 1000
        assert data["model"]["gamma"] == 1.0


def test_c1_dtd_single_factor_isolation():
    """Verify that P0, P1, P2 differ ONLY in model.projection_init."""
    p0 = yaml.safe_load((CONFIGS_DIR / "c1_p0_gaussian_std1.yaml").read_text(encoding="utf-8"))
    p1 = yaml.safe_load((CONFIGS_DIR / "c1_p1_official_uniform.yaml").read_text(encoding="utf-8"))
    p2 = yaml.safe_load((CONFIGS_DIR / "c1_p2_gaussian_matched_variance.yaml").read_text(encoding="utf-8"))

    assert p0["model"]["projection_init"] == "gaussian_std1"
    assert p1["model"]["projection_init"] == "official_uniform"
    assert p2["model"]["projection_init"] == "gaussian_matched_variance"

    # Deep compare with projection_init neutralized
    p0_copy = dict(p0)
    p1_copy = dict(p1)
    p2_copy = dict(p2)

    for cfg in (p0_copy, p1_copy, p2_copy):
        cfg["model"]["projection_init"] = "MATCHED"
        cfg["logging"]["output_dir"] = "MATCHED"
        cfg["logging"]["run_name"] = "MATCHED"
        cfg["logging"]["tags"] = ["MATCHED"]

    assert p0_copy == p1_copy
    assert p1_copy == p2_copy


def test_c1_dtd_chaining_identity_p1_equals_c2_b0():
    """Verify C1-P1 and C2-B0 are completely identical in model, dataset, and training params."""
    p1 = yaml.safe_load((CONFIGS_DIR / "c1_p1_official_uniform.yaml").read_text(encoding="utf-8"))
    b0 = yaml.safe_load((CONFIGS_DIR / "c2_b0_augreg_in1k.yaml").read_text(encoding="utf-8"))

    assert p1["dataset"] == b0["dataset"]
    assert p1["model"] == b0["model"]
    assert p1["training"] == b0["training"]
    assert p1["seed"] == b0["seed"]
    assert p1["class_order_seed"] == b0["class_order_seed"]
    assert p1["projection_seed"] == b0["projection_seed"]
    assert p1["stream_seed"] == b0["stream_seed"]


def test_c1_projection_variance_scaling_mathematics():
    """Test theoretical variance and std bounds across projection inits."""
    d, D = 768, 1000
    expected_matched_std = 1.0 / math.sqrt(3.0 * d)  # ~0.0208333

    proj_u = SmoothProjection(d, D, initialization="official_uniform", seed=42)
    meta_u = proj_u.get_metadata()
    assert math.isclose(meta_u["theoretical_std"], expected_matched_std, rel_tol=1e-5)
    assert math.isclose(meta_u["measured_std"], expected_matched_std, rel_tol=0.05)

    proj_g = SmoothProjection(d, D, initialization="gaussian_matched_variance", seed=42)
    meta_g = proj_g.get_metadata()
    assert math.isclose(meta_g["theoretical_std"], expected_matched_std, rel_tol=1e-5)
    assert math.isclose(meta_g["measured_std"], expected_matched_std, rel_tol=0.05)

    proj_0 = SmoothProjection(d, D, initialization="gaussian_std1", seed=42)
    meta_0 = proj_0.get_metadata()
    assert meta_0["theoretical_std"] == 1.0
    assert math.isclose(meta_0["measured_std"], 1.0, rel_tol=0.05)


def test_compare_c1_projections_on_synthetic_features():
    """Verify compare_c1_projections_on_features returns expected metrics without error."""
    torch.manual_seed(42)
    features = torch.randn(50, 768) * 1.5

    result = compare_c1_projections_on_features(
        features=features,
        projection_dim=1000,
        projection_seed=42,
        gamma=1.0,
    )
    assert "configurations" in result
    assert "gaussian_std1" in result["configurations"]
    assert "official_uniform" in result["configurations"]
    assert "gaussian_matched_variance" in result["configurations"]

    p0 = result["configurations"]["gaussian_std1"]
    p1 = result["configurations"]["official_uniform"]

    # P0 with std=1 on features with norm ~20 should have much higher saturation than P1
    assert p0["pre_sigmoid_u"]["std"] > p1["pre_sigmoid_u"]["std"] * 5.0
