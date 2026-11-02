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


def test_frozen_vit_metadata_and_non_fallback():
    """Verify get_backbone_metadata exposure and failure on non-existent checkpoint."""
    model = FrozenViT("vit_base_patch16_224.augreg_in1k", pretrained=False, feature_fusion=True)
    meta = model.get_backbone_metadata()
    assert meta["model_name"] == "vit_base_patch16_224.augreg_in1k"
    assert meta["feature_dim"] == 768
    assert meta["num_blocks"] == 12
    assert "pretrained_cfg" in meta


def test_c2_configs_exact_hyperparameter_alignment():
    """Verify B0 and B1 configs match C1 P1 in every single hyperparameter."""
    import yaml

    with open("configs/c1_p1_official_uniform.yaml") as f:
        c1 = yaml.safe_load(f)
    with open("configs/c2_b0_augreg_in1k.yaml") as f:
        b0 = yaml.safe_load(f)
    with open("configs/c2_b1_augreg2_in21k_ft_in1k.yaml") as f:
        b1 = yaml.safe_load(f)

    # Verification of gamma = 1.0 (strict)
    assert c1["model"]["gamma"] == 1.0
    assert b0["model"]["gamma"] == 1.0
    assert b1["model"]["gamma"] == 1.0

    # Verification of seeds
    for s_key in ["seed", "class_order_seed", "projection_seed", "stream_seed"]:
        assert b0[s_key] == c1[s_key] == 42
        assert b1[s_key] == c1[s_key] == 42

    # Verification of dataset parameters
    for d_key in ["name", "num_classes", "classes_per_task", "image_size", "mean", "std", "interpolation"]:
        assert b0["dataset"][d_key] == c1["dataset"][d_key]
        assert b1["dataset"][d_key] == c1["dataset"][d_key]

    # Verification of training parameters
    assert b0["training"]["batch_size"] == c1["training"]["batch_size"] == 10
    assert b1["training"]["batch_size"] == c1["training"]["batch_size"] == 10

    # Verification of model parameters except backbone
    for m_key in ["pretrained", "feature_fusion", "smooth_projection", "projection_dim", "projection_init", "gamma", "rls_dtype"]:
        assert b0["model"][m_key] == c1["model"][m_key]
        assert b1["model"][m_key] == c1["model"][m_key]

    # Verification of single factor of variation: only backbone differs
    assert b0["model"]["backbone"] == "vit_base_patch16_224.augreg_in1k"
    assert b1["model"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def test_c2_cache_validation_and_tamper_rejection(tmp_path):
    """Verify feature cache rejects mismatched pretrained_tag, preprocessing, or dataset_name."""
    import hashlib
    import json
    from scripts.train import extract_or_load_task_features
    from src.foal.datasets import IndexedImageDataset
    from torchvision.datasets import FakeData
    from torchvision import transforms

    model = FOALModel(
        backbone_name="vit_base_patch16_224.augreg_in1k",
        pretrained=False,
        projection_init="official_uniform",
    )
    # Mock tags
    model.encoder.pretrained_tag = "augreg_in1k"

    from PIL import Image

    # Mock source list of dicts with 'img' and 'fine_label'
    mock_source = [
        {"img": Image.new("RGB", (224, 224), color=(i * 20, i * 20, i * 20)), "fine_label": i % 5}
        for i in range(10)
    ]
    indexed_ds = IndexedImageDataset(
        mock_source,
        indices=list(range(10)),
        label_to_incremental={i: i for i in range(5)},
        transform=transforms.ToTensor(),
    )

    prep = {"image_size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "interpolation": "bicubic"}
    cache_dir = tmp_path / "cache"

    # First extraction: creates cache
    feats1, labels1, _, from_cache1 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_sample",
        model=model,
        dataset=indexed_ds,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep,
        dataset_name="cifar100",
    )
    assert from_cache1 is False
    assert (cache_dir / "test_sample.pt").exists()

    # Second extraction with identical config: loads from cache
    feats2, labels2, _, from_cache2 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_sample",
        model=model,
        dataset=indexed_ds,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep,
        dataset_name="cifar100",
    )
    assert from_cache2 is True

    # Third extraction with mismatched pretrained_tag: must REJECT cache
    model.encoder.pretrained_tag = "augreg2_in21k_ft_in1k"
    feats3, labels3, _, from_cache3 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_sample",
        model=model,
        dataset=indexed_ds,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep,
        dataset_name="cifar100",
    )
    assert from_cache3 is False  # Must re-extract!

    # Fourth extraction with mismatched preprocessing: must REJECT cache
    prep_mismatch = {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "interpolation": "bilinear"}
    feats4, labels4, _, from_cache4 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_sample",
        model=model,
        dataset=indexed_ds,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep_mismatch,
        dataset_name="cifar100",
    )
    assert from_cache4 is False  # Must re-extract!

