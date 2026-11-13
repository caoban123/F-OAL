"""Unit and regression tests for Experiment C2 on DTD (Backbone Alignment)."""
from __future__ import annotations

from pathlib import Path
import pytest
import timm
import torch
import yaml

from src.foal.projection import SmoothProjection
from src.foal.diagnostics import compare_c2_backbone_conditions


CONFIGS_DIR = Path("configs/dtd")


def test_c2_dtd_configs_exist_and_parse():
    """Verify both C2 DTD configs exist and parse valid YAML."""
    c2_files = [
        CONFIGS_DIR / "c2_b0_augreg_in1k.yaml",
        CONFIGS_DIR / "c2_b1_augreg2_in21k_ft_in1k.yaml",
    ]
    for p in c2_files:
        assert p.exists(), f"Missing config: {p}"
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert data["dataset"]["type"] == "dtd"
        assert data["dataset"]["name"] == "tanganke/dtd"
        assert data["dataset"]["samples_per_class"] == 40
        assert data["model"]["projection_init"] == "official_uniform"
        assert data["model"]["projection_dim"] == 1000
        assert data["model"]["gamma"] == 1.0


def test_c2_dtd_single_factor_isolation():
    """Verify B0 and B1 differ ONLY in model.backbone."""
    b0 = yaml.safe_load((CONFIGS_DIR / "c2_b0_augreg_in1k.yaml").read_text(encoding="utf-8"))
    b1 = yaml.safe_load((CONFIGS_DIR / "c2_b1_augreg2_in21k_ft_in1k.yaml").read_text(encoding="utf-8"))

    assert b0["model"]["backbone"] == "vit_base_patch16_224.augreg_in1k"
    assert b1["model"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"

    b0_copy = dict(b0)
    b1_copy = dict(b1)
    b0_copy["model"]["backbone"] = "MATCHED"
    b1_copy["model"]["backbone"] = "MATCHED"
    b0_copy["logging"]["output_dir"] = "MATCHED"
    b1_copy["logging"]["output_dir"] = "MATCHED"
    b0_copy["logging"]["run_name"] = "MATCHED"
    b1_copy["logging"]["run_name"] = "MATCHED"
    b0_copy["logging"]["tags"] = ["MATCHED"]
    b1_copy["logging"]["tags"] = ["MATCHED"]

    assert b0_copy == b1_copy


def test_c2_dtd_chaining_identity_b1_equals_c3_a():
    """Verify C2-B1 matches C3-A in dataset, model, training, and seed settings."""
    b1 = yaml.safe_load((CONFIGS_DIR / "c2_b1_augreg2_in21k_ft_in1k.yaml").read_text(encoding="utf-8"))
    c3_a = yaml.safe_load((CONFIGS_DIR / "c3_a_bicubic_normalized.yaml").read_text(encoding="utf-8"))

    assert b1["dataset"] == c3_a["dataset"]
    assert b1["model"] == c3_a["model"]
    assert b1["training"] == c3_a["training"]
    assert b1["seed"] == c3_a["seed"]
    assert b1["class_order_seed"] == c3_a["class_order_seed"]
    assert b1["projection_seed"] == c3_a["projection_seed"]
    assert b1["stream_seed"] == c3_a["stream_seed"]


def test_c2_dtd_projection_equality():
    """Verify identical projection matrix across B0 and B1 under seed 42."""
    p_b0 = SmoothProjection(768, 1000, initialization="official_uniform", seed=42)
    p_b1 = SmoothProjection(768, 1000, initialization="official_uniform", seed=42)

    assert torch.equal(p_b0.weight, p_b1.weight)
    meta_b0 = p_b0.get_metadata()
    meta_b1 = p_b1.get_metadata()
    assert meta_b0["matrix_sha256"] == meta_b1["matrix_sha256"]


def test_c2_dtd_backbones_valid():
    """Verify timm model names resolve correctly."""
    b0 = "vit_base_patch16_224.augreg_in1k"
    b1 = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    assert timm.is_model(b0)
    assert timm.is_model(b1)


def test_compare_c2_diagnostics_on_synthetic_features():
    """Verify compare_c2_backbone_conditions on synthetic features."""
    torch.manual_seed(42)
    b0_f = torch.randn(40, 768) * 1.0
    b1_f = torch.randn(40, 768) * 1.5
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)

    report = compare_c2_backbone_conditions(
        b0_features=b0_f,
        b1_features=b1_f,
        labels=labels,
        projection_dim=1000,
        projection_seed=42,
        gamma=1.0,
    )
    assert "b0_augreg_in1k" in report
    assert "b1_augreg2_in21k_ft_in1k" in report
    assert report["projection_equality_verified"] is True
    assert report["c2_diagnostics_verified"] is True
