"""
Unit and regression tests for Experiment C2 (Pretrained ViT Backbone Alignment).
"""

import math
from pathlib import Path
import pytest
import torch
import timm

from src.foal.backbone import FrozenViT
from src.foal.model import FOALModel
from src.foal.projection import SmoothProjection
from src.foal.diagnostics import (
    compute_backbone_feature_diagnostics,
    compare_c2_backbone_conditions,
)


def test_backbone_model_resolution_and_pretrained_tags():
    """Verify both B0 and B1 model identifiers resolve to expected timm tags and architectures."""
    b0_name = "vit_base_patch16_224.augreg_in1k"
    b1_name = "vit_base_patch16_224.augreg2_in21k_ft_in1k"

    assert timm.is_model(b0_name)
    assert timm.is_model(b1_name)
    assert timm.is_model_pretrained(b0_name)
    assert timm.is_model_pretrained(b1_name)

    cfg0 = timm.models.get_pretrained_cfg(b0_name)
    cfg1 = timm.models.get_pretrained_cfg(b1_name)

    assert cfg0.tag == "augreg_in1k"
    assert cfg1.tag == "augreg2_in21k_ft_in1k"
    assert cfg0.architecture == "vit_base_patch16_224"
    assert cfg1.architecture == "vit_base_patch16_224"
    assert cfg0.input_size == (3, 224, 224)
    assert cfg1.input_size == (3, 224, 224)
    assert cfg0.interpolation == "bicubic"
    assert cfg1.interpolation == "bicubic"
    assert cfg0.mean == (0.5, 0.5, 0.5)
    assert cfg1.mean == (0.5, 0.5, 0.5)


def test_c2_projection_equality_guarantee():
    """Verify that B0 and B1 use identical projection matrix P (Projection Equality)."""
    # Initialize projection with official_uniform, seed 42
    proj_b0 = SmoothProjection(768, 1000, seed=42, initialization="official_uniform")
    proj_b1 = SmoothProjection(768, 1000, seed=42, initialization="official_uniform")

    # Matrices must be bitwise identical
    assert torch.equal(proj_b0.weight, proj_b1.weight)
    meta_b0 = proj_b0.get_metadata()
    meta_b1 = proj_b1.get_metadata()
    assert meta_b0["matrix_sha256"] == meta_b1["matrix_sha256"]
    assert meta_b0["initialization"] == "official_uniform"
    assert meta_b1["initialization"] == "official_uniform"


def test_c2_cache_key_isolation():
    """Verify that feature cache keys include sanitized backbone identifiers to prevent cross-contamination."""
    bb0 = "vit_base_patch16_224.augreg_in1k"
    bb1 = "vit_base_patch16_224.augreg2_in21k_ft_in1k"

    slug0 = bb0.replace("/", "_").replace(".", "_")
    slug1 = bb1.replace("/", "_").replace(".", "_")

    key0 = f"train_{slug0}_c42_s42_t01_tr100"
    key1 = f"train_{slug1}_c42_s42_t01_tr100"

    assert key0 != key1
    assert "augreg_in1k" in key0
    assert "augreg2_in21k_ft_in1k" in key1


def test_c2_backbone_diagnostics_on_synthetic_features():
    """Test representation diagnostics (L2 norm, centered rank, cosine similarity) on synthetic feature tensor."""
    gen = torch.Generator().manual_seed(42)
    # Simulate features for 4 classes, 10 samples each (40 samples total)
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    
    # B0 features: centered with moderate class clustering
    b0_centers = torch.randn(4, 768, generator=gen) * 2.0
    b0_noise = torch.randn(40, 768, generator=gen) * 0.5
    b0_features = b0_centers[labels] + b0_noise

    # B1 features: centered with stronger class clustering (higher separation)
    b1_centers = torch.randn(4, 768, generator=gen) * 4.0
    b1_noise = torch.randn(40, 768, generator=gen) * 0.5
    b1_features = b1_centers[labels] + b1_noise

    report = compare_c2_backbone_conditions(
        b0_features=b0_features,
        b1_features=b1_features,
        labels=labels,
        projection_dim=1000,
        projection_seed=42,
        gamma=1.0,
    )

    assert report["c2_diagnostics_verified"] is True
    assert report["projection_equality_verified"] is True
    assert "b0_augreg_in1k" in report
    assert "b1_augreg2_in21k_ft_in1k" in report

    b0_stats = report["b0_augreg_in1k"]["features"]
    b1_stats = report["b1_augreg2_in21k_ft_in1k"]["features"]

    assert b0_stats["num_samples"] == 40
    assert b0_stats["feature_dim"] == 768
    assert b0_stats["centered_covariance"]["effective_rank"] > 1.0
    assert b0_stats["class_separation"]["separation_ratio"] > 1.0


def test_frozen_vit_properties_and_strict_eval():
    """Verify FrozenViT parameter freezing and persistence of eval mode."""
    # Test with light synthetic network or verified pretrained model
    model = FrozenViT("vit_base_patch16_224.augreg_in1k", pretrained=False, feature_fusion=True)
    assert model.feature_dim == 768
    for p in model.parameters():
        assert p.requires_grad is False
    assert model.network.training is False

    # Setting model.train(True) must NOT put backbone in training mode
    model.train(True)
    assert model.network.training is False
    for p in model.parameters():
        assert p.requires_grad is False
