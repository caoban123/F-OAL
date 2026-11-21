"""Unit and regression tests for Experiment C3 on DTD (2x2 Preprocessing Alignment)."""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image
import pytest
import torch
import torchvision.transforms as tv_transforms
import yaml

from src.foal.datasets import cifar100_transforms


CONFIGS_DIR = Path("configs/dtd")


def test_c3_dtd_configs_exist_and_parse():
    """Verify all 4 C3 DTD configs exist and parse valid YAML."""
    c3_files = [
        CONFIGS_DIR / "c3_a_bicubic_normalized.yaml",
        CONFIGS_DIR / "c3_b_bilinear_normalized.yaml",
        CONFIGS_DIR / "c3_c_bicubic_raw.yaml",
        CONFIGS_DIR / "c3_d_bilinear_raw.yaml",
    ]
    for p in c3_files:
        assert p.exists(), f"Missing config: {p}"
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert data["dataset"]["type"] == "dtd"
        assert data["dataset"]["name"] == "tanganke/dtd"
        assert data["dataset"]["samples_per_class"] == 40
        assert data["model"]["backbone"] == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
        assert data["model"]["projection_init"] == "official_uniform"


def test_c3_dtd_2x2_factorial_isolation():
    """Verify 2x2 factorial structure: Interp (bicubic vs bilinear) x Norm (true vs false)."""
    cfgs = {
        "A": yaml.safe_load((CONFIGS_DIR / "c3_a_bicubic_normalized.yaml").read_text(encoding="utf-8")),
        "B": yaml.safe_load((CONFIGS_DIR / "c3_b_bilinear_normalized.yaml").read_text(encoding="utf-8")),
        "C": yaml.safe_load((CONFIGS_DIR / "c3_c_bicubic_raw.yaml").read_text(encoding="utf-8")),
        "D": yaml.safe_load((CONFIGS_DIR / "c3_d_bilinear_raw.yaml").read_text(encoding="utf-8")),
    }

    assert cfgs["A"]["dataset"]["interpolation"] == "bicubic"
    assert cfgs["A"]["dataset"]["normalize"] is True

    assert cfgs["B"]["dataset"]["interpolation"] == "bilinear"
    assert cfgs["B"]["dataset"]["normalize"] is True

    assert cfgs["C"]["dataset"]["interpolation"] == "bicubic"
    assert cfgs["C"]["dataset"]["normalize"] is False

    assert cfgs["D"]["dataset"]["interpolation"] == "bilinear"
    assert cfgs["D"]["dataset"]["normalize"] is False

    # All non-preprocessing fields must match identically across all 4
    for key in ("B", "C", "D"):
        assert cfgs["A"]["dataset"]["name"] == cfgs[key]["dataset"]["name"]
        assert cfgs["A"]["dataset"]["samples_per_class"] == cfgs[key]["dataset"]["samples_per_class"]
        assert cfgs["A"]["model"] == cfgs[key]["model"]
        assert cfgs["A"]["seed"] == cfgs[key]["seed"]
        assert cfgs["A"]["class_order_seed"] == cfgs[key]["class_order_seed"]
        assert cfgs["A"]["projection_seed"] == cfgs[key]["projection_seed"]
        assert cfgs["A"]["stream_seed"] == cfgs[key]["stream_seed"]


def test_c3_dtd_c3_d_matches_table4_a0():
    """Verify C3-D matches configs/dtd/dtd_a0_full.yaml in model, dataset, and training parameters."""
    c3_d = yaml.safe_load((CONFIGS_DIR / "c3_d_bilinear_raw.yaml").read_text(encoding="utf-8"))
    a0 = yaml.safe_load((CONFIGS_DIR / "dtd_a0_full.yaml").read_text(encoding="utf-8"))

    assert c3_d["dataset"]["name"] == a0["dataset"]["name"]
    assert c3_d["dataset"]["type"] == a0["dataset"]["type"]
    assert c3_d["dataset"]["interpolation"] == a0["dataset"]["interpolation"]
    assert c3_d["dataset"]["normalize"] == a0["dataset"]["normalize"]
    assert c3_d["dataset"]["samples_per_class"] == a0["dataset"]["samples_per_class"]
    assert c3_d["model"]["backbone"] == a0["model"]["backbone"]
    assert c3_d["model"]["projection_init"] == a0["model"]["projection_init"]
    assert c3_d["model"]["gamma"] == a0["model"]["gamma"]
    assert c3_d["seed"] == a0["seed"]


def test_c3_d_bitwise_equivalence_on_non_square_dtd_images():
    """Verify C3-D transform produces identical output to Official Author transform on non-square images."""
    official_pipeline = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
    ])
    c3_d_pipeline = cifar100_transforms(size=224, interpolation="bilinear", normalize=False)

    test_sizes = [(300, 200), (640, 480), (128, 256), (333, 333)]
    for w, h in test_sizes:
        img = Image.new("RGB", (w, h), color=(100, 150, 200))
        t_off = official_pipeline(img)
        t_c3d = c3_d_pipeline(img)
        max_diff = (t_off - t_c3d).abs().max().item()
        assert max_diff < 1e-6, f"Discrepancy on size {(w, h)}: {max_diff}"
        assert t_c3d.shape == (3, 224, 224)
        assert t_c3d.min() >= 0.0 and t_c3d.max() <= 1.0


def test_2x2_factorial_math_identities():
    """Verify the factorial contrast identity: Total == Interp + Norm."""
    # Arbitrary test numbers for A, B, C, D
    a, b, c, d = 82.5, 84.0, 83.0, 86.0
    delta_interp = 0.5 * ((b - a) + (d - c))
    delta_norm = 0.5 * ((c - a) + (d - b))
    total_effect = d - a

    assert math.isclose(total_effect, delta_interp + delta_norm, abs_tol=1e-9)
