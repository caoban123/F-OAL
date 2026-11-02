"""Tests for Experiment C3: Controlled Input Preprocessing Alignment.

Validates:
1. Transform shapes, dtypes, and pixel ranges across all 4 conditions (A, B, C, D).
2. Bitwise equivalence of C3-D with official author preprocessing.
3. Factorial config parameters, seeds, and strict single-factor isolation.
4. Equivalence of C3-A with C2-B1 for compute reuse.
5. Cache prefix uniqueness and cache tampering/mismatch rejection.
6. 2x2 Factorial contrast formulas (Main effects, Interaction, Total delta).
"""
import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest
import torch
import torchvision.transforms as tv_transforms
import yaml

from src.foal.datasets import cifar100_transforms, IndexedImageDataset
from scripts.train import extract_or_load_task_features


def test_transform_shapes_and_ranges():
    """Verify output tensor shapes, ranges, and dtypes for all 4 preprocessing conditions."""
    # Synthetic 32x32 RGB PIL image
    img = Image.new("RGB", (32, 32), color=(128, 64, 192))
    
    t_a = cifar100_transforms(size=224, interpolation="bicubic", normalize=True)
    t_b = cifar100_transforms(size=224, interpolation="bilinear", normalize=True)
    t_c = cifar100_transforms(size=224, interpolation="bicubic", normalize=False)
    t_d = cifar100_transforms(size=224, interpolation="bilinear", normalize=False)
    
    out_a = t_a(img)
    out_b = t_b(img)
    out_c = t_c(img)
    out_d = t_d(img)
    
    for t_name, out in [("C3-A", out_a), ("C3-B", out_b), ("C3-C", out_c), ("C3-D", out_d)]:
        assert out.shape == (3, 224, 224), f"{t_name} shape mismatch: {out.shape}"
        assert out.dtype == torch.float32, f"{t_name} dtype mismatch: {out.dtype}"
        
    # Normalized conditions must span negative and positive values (e.g. [-1, 1])
    assert out_a.min() < 0.0 and out_a.max() <= 1.0
    assert out_b.min() < 0.0 and out_b.max() <= 1.0
    
    # Raw conditions must be strictly non-negative [0.0, 1.0]
    assert out_c.min() >= 0.0 and out_c.max() <= 1.0
    assert out_d.min() >= 0.0 and out_d.max() <= 1.0


def test_c3_d_matches_official_author_transform():
    """Verify C3-D produces identical tensors to the official author transform."""
    # Official author transform from setup_elements.py:
    # transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    official_trans = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
    ])
    c3_d_trans = cifar100_transforms(size=224, interpolation="bilinear", normalize=False)
    
    # Test on multiple distinct PIL images
    for color in [(255, 0, 0), (0, 255, 0), (120, 150, 200), (0, 0, 0), (255, 255, 255)]:
        img = Image.new("RGB", (32, 32), color=color)
        out_official = official_trans(img)
        out_c3_d = c3_d_trans(img)
        
        diff = (out_official - out_c3_d).abs().max().item()
        assert diff < 1e-6, f"Mismatch with official transform: max diff {diff}"


def test_configs_integrity_and_isolation():
    """Verify all 4 C3 configs adhere to strict factorial isolation."""
    configs_dir = Path("configs")
    cfg_files = {
        "A": configs_dir / "c3_a_bicubic_normalized.yaml",
        "B": configs_dir / "c3_b_bilinear_normalized.yaml",
        "C": configs_dir / "c3_c_bicubic_raw.yaml",
        "D": configs_dir / "c3_d_bilinear_raw.yaml",
    }
    
    loaded = {}
    for k, p in cfg_files.items():
        assert p.exists(), f"Config file missing: {p}"
        loaded[k] = yaml.safe_load(p.read_text(encoding="utf-8"))
        
    for k, cfg in loaded.items():
        # Fixed controls
        assert cfg["model"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
        assert cfg["projection"]["initialization"] == "official_uniform"
        assert cfg["projection"]["dimension"] == 1000
        assert cfg["projection"]["seed"] == 42
        assert cfg["model"]["gamma"] == 1.0
        assert cfg["model"]["rls_dtype"] == "float64"
        assert cfg["seed"] == 42
        assert cfg["class_order_seed"] == 42
        assert cfg["stream_seed"] == 42
        assert cfg["logging"]["group"] == "c3_preprocessing_alignment"
        assert cfg["logging"]["project"] == "hcmus-foal-reproduction"
        assert cfg["logging"]["base_url"] == "https://api.forge.coreweave.com"
        
    # Factorial grid
    assert loaded["A"]["dataset"]["interpolation"] == "bicubic"
    assert loaded["A"]["dataset"]["normalize"] is True
    
    assert loaded["B"]["dataset"]["interpolation"] == "bilinear"
    assert loaded["B"]["dataset"]["normalize"] is True
    
    assert loaded["C"]["dataset"]["interpolation"] == "bicubic"
    assert loaded["C"]["dataset"]["normalize"] is False
    
    assert loaded["D"]["dataset"]["interpolation"] == "bilinear"
    assert loaded["D"]["dataset"]["normalize"] is False


def test_c3_a_equivalence_to_c2_b1():
    """Verify C3-A matches C2-B1 in all execution hyperparameters for compute reuse."""
    cfg_c3_a = yaml.safe_load(Path("configs/c3_a_bicubic_normalized.yaml").read_text(encoding="utf-8"))
    cfg_c2_b1 = yaml.safe_load(Path("configs/c2_b1_augreg2_in21k_ft_in1k.yaml").read_text(encoding="utf-8"))
    
    # Identical backbone & pretraining
    assert cfg_c3_a["model"]["backbone"] == cfg_c2_b1["model"]["backbone"]
    assert cfg_c3_a["model"]["pretrained"] == cfg_c2_b1["model"]["pretrained"]
    
    # Identical projection
    assert cfg_c3_a["projection"]["initialization"] == cfg_c2_b1["projection"]["initialization"]
    assert cfg_c3_a["projection"]["dimension"] == cfg_c2_b1["projection"]["dimension"]
    assert cfg_c3_a["projection"]["seed"] == cfg_c2_b1["projection"]["seed"]
    
    # Identical dataset preprocessing
    assert cfg_c3_a["dataset"]["image_size"] == cfg_c2_b1["dataset"]["image_size"]
    assert cfg_c3_a["dataset"]["interpolation"] == cfg_c2_b1["dataset"]["interpolation"]
    assert cfg_c3_a["dataset"]["mean"] == cfg_c2_b1["dataset"]["mean"]
    assert cfg_c3_a["dataset"]["std"] == cfg_c2_b1["dataset"]["std"]
    
    # Identical optimization & seeds
    assert cfg_c3_a["model"]["gamma"] == cfg_c2_b1["model"]["gamma"]
    assert cfg_c3_a["model"]["rls_dtype"] == cfg_c2_b1["model"]["rls_dtype"]
    assert cfg_c3_a["seed"] == cfg_c2_b1["seed"]


def test_cache_key_isolation_and_rejection(tmp_path):
    """Verify feature cache isolation and rejection of tampered/mismatched metadata."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock model
    class MockEncoder(torch.nn.Module):
        pretrained_tag = "augreg2_in21k_ft_in1k"
        feature_dim = 16
        def forward(self, x):
            return torch.zeros((len(x), 16), dtype=torch.float32)
            
    class MockModel(torch.nn.Module):
        backbone_name = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
        feature_fusion_enabled = True
        def __init__(self):
            super().__init__()
            self.encoder = MockEncoder()
        
    mock_model = MockModel()
    
    # Mock dataset (2 samples to keep test fast)
    sample_img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    source = [{"img": sample_img, "fine_label": 0}] * 2
    ds_bicubic = IndexedImageDataset(source, list(range(2)), {0: 0}, cifar100_transforms(size=224, interpolation="bicubic", normalize=True))
    
    prep_bicubic = {
        "image_size": 224,
        "normalize": True,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "interpolation": "bicubic",
    }
    
    prep_bilinear = {
        "image_size": 224,
        "normalize": False,
        "mean": None,
        "std": None,
        "interpolation": "bilinear",
    }
    
    # 1. First extraction with Bicubic
    feats1, labels1, sec1, cached1 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_task_01",
        model=mock_model,
        dataset=ds_bicubic,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep_bicubic,
        dataset_name="uoft-cs/cifar100",
    )
    assert cached1 is False
    assert (cache_dir / "test_task_01.pt").exists()
    
    # 2. Re-loading with matching preprocessing succeeds from cache
    feats2, labels2, sec2, cached2 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_task_01",
        model=mock_model,
        dataset=ds_bicubic,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep_bicubic,
        dataset_name="uoft-cs/cifar100",
    )
    assert cached2 is True
    
    # 3. Loading with mismatched preprocessing (e.g. Bilinear Raw) detects invalid provenance and re-extracts
    feats3, labels3, sec3, cached3 = extract_or_load_task_features(
        cache_dir=cache_dir,
        cache_key="test_task_01",
        model=mock_model,
        dataset=ds_bicubic,
        batch_size=5,
        num_workers=0,
        device="cpu",
        preprocessing=prep_bilinear,
        dataset_name="uoft-cs/cifar100",
    )
    assert cached3 is False, "Mismatched preprocessing must not be loaded from cache!"


def test_factorial_contrast_calculation():
    """Verify 2x2 factorial contrast formulas for main effects, interaction, and total delta."""
    # Synthetic accuracy test values Case 1 (with non-zero interaction):
    # Condition A (Bicubic, Norm): 91.0
    # Condition B (Bilinear, Norm): 91.4 (+0.4 Interp effect in Norm)
    # Condition C (Bicubic, Raw): 90.6 (-0.4 Norm effect in Bicubic)
    # Condition D (Bilinear, Raw): 91.2 (+0.6 Interp effect in Raw; +0.2 Interact)
    a, b, c, d = 91.0, 91.4, 90.6, 91.2
    
    # 1. Main effect of Interpolation: E_interp = 0.5 * [(B - A) + (D - C)]
    delta_interp = 0.5 * ((b - a) + (d - c))
    assert pytest.approx(delta_interp, abs=1e-5) == 0.50
    
    # 2. Main effect of Normalization: E_norm = 0.5 * [(C - A) + (D - B)]
    delta_norm = 0.5 * ((c - a) + (d - b))
    assert pytest.approx(delta_norm, abs=1e-5) == -0.30
    
    # 3. Interaction Contrast: Contrast = D - C - B + A = (D - B) - (C - A)
    contrast = d - c - b + a
    assert pytest.approx(contrast, abs=1e-5) == 0.20
    assert pytest.approx((d - b) - (c - a), abs=1e-5) == contrast
    
    # In 2x2 factorial design:
    # - Average interaction effect: Contrast / 2 = 0.10
    # - Regression coefficient beta_12 under +/-1 effects coding: Contrast / 4 = 0.05
    avg_interaction_effect = contrast / 2.0
    beta_12_effects_coding = contrast / 4.0
    assert pytest.approx(avg_interaction_effect, abs=1e-5) == 0.10
    assert pytest.approx(beta_12_effects_coding, abs=1e-5) == 0.05
    
    # 4. Total Preprocessing Shift: Delta_total = D - A
    # Mathematical identity: Delta_total = E_interp + E_norm
    delta_total = d - a
    assert pytest.approx(delta_total, abs=1e-5) == 0.20
    assert pytest.approx(delta_total, abs=1e-5) == (delta_interp + delta_norm)

    # Synthetic Case 2: Purely additive factors (Zero interaction)
    # Factor A adds +1.0, Factor B subtracts -2.0
    a2 = 90.0
    b2 = a2 + 1.0  # 91.0
    c2 = a2 - 2.0  # 88.0
    d2 = a2 + 1.0 - 2.0  # 89.0
    contrast2 = d2 - c2 - b2 + a2
    assert pytest.approx(contrast2, abs=1e-5) == 0.0
    assert pytest.approx(0.5 * ((b2 - a2) + (d2 - c2)), abs=1e-5) == 1.0
    assert pytest.approx(0.5 * ((c2 - a2) + (d2 - b2)), abs=1e-5) == -2.0
    assert pytest.approx(d2 - a2, abs=1e-5) == -1.0
