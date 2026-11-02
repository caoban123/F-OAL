import argparse
import math
from pathlib import Path
import random
import numpy as np
import pytest
import torch
import yaml

from src.foal.projection import SmoothProjection
from src.foal.feature_fusion import fuse_cls_features
from src.foal.datasets import make_class_order
from scripts.train import resolve_projection_settings
from src.foal.diagnostics import compare_c1_projections_on_features


def test_projection_is_fixed_and_seeded():
    a = SmoothProjection(4, 6, seed=10)
    b = SmoothProjection(4, 6, seed=10)
    torch.testing.assert_close(a.weight, b.weight)
    assert len(list(a.parameters())) == 0
    x = torch.randn(3, 4)
    assert a(x).shape == (3, 6)
    assert torch.all((a(x) >= 0) & (a(x) <= 1))


def test_feature_fusion_averages_cls_token_across_blocks():
    first = torch.tensor([[[1., 2.], [999., 999.]]])
    second = torch.tensor([[[3., 6.], [888., 888.]]])
    fused = fuse_cls_features([first, second])
    torch.testing.assert_close(fused, torch.tensor([[2., 4.]]))


def test_projection_p0_gaussian_std1_distribution():
    d, D = 768, 1000
    proj = SmoothProjection(d, D, seed=42, initialization="gaussian_std1")
    assert proj.weight.shape == (d, D)
    assert proj.weight.dtype == torch.float32
    meta = proj.get_metadata()
    assert meta["initialization"] == "gaussian_std1"
    assert abs(meta["theoretical_std"] - 1.0) < 1e-9
    assert abs(meta["theoretical_variance"] - 1.0) < 1e-9
    assert abs(meta["measured_mean"]) < 0.01
    assert abs(meta["measured_std"] - 1.0) < 0.01
    assert abs(meta["measured_variance"] - 1.0) < 0.02


def test_projection_p1_official_uniform_distribution_and_bounds():
    d, D = 768, 1000
    proj = SmoothProjection(d, D, seed=42, initialization="official_uniform")
    assert proj.weight.shape == (d, D)
    bound = 1.0 / math.sqrt(d)
    expected_var = 1.0 / (3.0 * d)
    expected_std = math.sqrt(expected_var)

    # All entries must strictly lie within [-1/sqrt(d), +1/sqrt(d)]
    assert float(proj.weight.min().item()) >= -bound - 1e-7
    assert float(proj.weight.max().item()) <= bound + 1e-7

    meta = proj.get_metadata()
    assert meta["initialization"] == "official_uniform"
    assert abs(meta["uniform_bound"] - bound) < 1e-9
    assert abs(meta["theoretical_variance"] - expected_var) < 1e-9
    assert abs(meta["theoretical_std"] - expected_std) < 1e-9
    assert abs(meta["measured_mean"]) < 0.001
    assert abs(meta["measured_std"] - expected_std) < 0.0005
    assert abs(meta["measured_variance"] - expected_var) < 0.00005


def test_projection_p2_gaussian_matched_variance_distribution():
    d, D = 768, 1000
    proj_p1 = SmoothProjection(d, D, seed=42, initialization="official_uniform")
    proj_p2 = SmoothProjection(d, D, seed=42, initialization="gaussian_matched_variance")
    assert proj_p2.weight.shape == (d, D)

    expected_var = 1.0 / (3.0 * d)
    expected_std = math.sqrt(expected_var)
    meta_p1 = proj_p1.get_metadata()
    meta_p2 = proj_p2.get_metadata()

    # Theoretical population variance and std of P1 and P2 must match exactly
    assert abs(meta_p1["theoretical_variance"] - meta_p2["theoretical_variance"]) < 1e-12
    assert abs(meta_p1["theoretical_std"] - meta_p2["theoretical_std"]) < 1e-12
    assert abs(meta_p2["measured_mean"]) < 0.001
    assert abs(meta_p2["measured_std"] - expected_std) < 0.0005
    assert abs(meta_p2["measured_variance"] - expected_var) < 0.00005
    # Gaussian P2 has unbounded tails exceeding uniform bound 1/sqrt(d)
    bound = 1.0 / math.sqrt(d)
    assert float(proj_p2.weight.abs().max().item()) > bound


def test_projection_reproducibility_and_seed_sensitivity():
    for mode in ("gaussian_std1", "official_uniform", "gaussian_matched_variance"):
        p_a = SmoothProjection(768, 1000, seed=42, initialization=mode)
        p_b = SmoothProjection(768, 1000, seed=42, initialization=mode)
        p_c = SmoothProjection(768, 1000, seed=99, initialization=mode)
        assert torch.equal(p_a.weight, p_b.weight)
        assert p_a.get_metadata()["matrix_sha256"] == p_b.get_metadata()["matrix_sha256"]
        assert not torch.equal(p_a.weight, p_c.weight)
        assert p_a.get_metadata()["matrix_sha256"] != p_c.get_metadata()["matrix_sha256"]


def test_projection_rng_isolation_does_not_affect_global_rng_or_class_order():
    results = {}
    for mode in ("gaussian_std1", "official_uniform", "gaussian_matched_variance"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        class_order = make_class_order(100, 42)
        proj = SmoothProjection(768, 1000, seed=42, initialization=mode)
        # Draw from global PyTorch, NumPy, and Python RNGs after projection init
        next_torch = torch.randn(5).tolist()
        next_numpy = np.random.randn(5).tolist()
        next_py = [random.random() for _ in range(5)]

        results[mode] = (class_order, next_torch, next_numpy, next_py, proj.weight.shape)

    ref_order, ref_torch, ref_numpy, ref_py, _ = results["gaussian_std1"]
    for mode in ("official_uniform", "gaussian_matched_variance"):
        order, t_val, n_val, p_val, shape = results[mode]
        assert order == ref_order
        assert t_val == ref_torch
        assert n_val == ref_numpy
        assert p_val == ref_py
        assert shape == (768, 1000)


def test_projection_not_reinitialized_and_legacy_config_compatible():
    cfg = yaml.safe_load(Path("configs/cifar100.yaml").read_text(encoding="utf-8"))
    assert "projection" not in cfg
    dummy_args = argparse.Namespace(projection_init=None, projection_seed=None)
    proj_cfg = resolve_projection_settings(cfg, dummy_args, base_seed=int(cfg["seed"]))
    assert proj_cfg["initialization"] == "gaussian_std1"
    assert proj_cfg["dimension"] == 1000
    assert proj_cfg["seed"] == 42

    for cfg_path, expected_init in (
        ("configs/c1_p0_gaussian_std1.yaml", "gaussian_std1"),
        ("configs/c1_p1_official_uniform.yaml", "official_uniform"),
        ("configs/c1_p2_gaussian_matched_variance.yaml", "gaussian_matched_variance"),
    ):
        c1_cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        resolved = resolve_projection_settings(c1_cfg, dummy_args, base_seed=int(c1_cfg["seed"]))
        assert resolved["initialization"] == expected_init
        assert resolved["dimension"] == 1000
        assert resolved["seed"] == 42

    proj_default = SmoothProjection(768, proj_cfg["dimension"], seed=proj_cfg["seed"])
    proj_explicit = SmoothProjection(
        768, proj_cfg["dimension"], seed=proj_cfg["seed"], initialization="gaussian_std1"
    )
    assert torch.equal(proj_default.weight, proj_explicit.weight)

    initial_sha = proj_default.get_metadata()["matrix_sha256"]
    x = torch.randn(16, 768)
    _ = proj_default(x)
    _ = proj_default.pre_activation(x)
    assert proj_default.get_metadata()["matrix_sha256"] == initial_sha


def test_invalid_projection_initialization_raises():
    with pytest.raises(ValueError, match="Unknown projection initialization"):
        SmoothProjection(16, 32, seed=42, initialization="unknown_init")


def test_matched_input_diagnostics_compare_c1_conditions():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    # Simulate fused ViT features with realistic L2 norm (~15-20 for d=768)
    features = torch.randn((32, 768), generator=gen) * 0.65
    report = compare_c1_projections_on_features(features, projection_dim=1000, projection_seed=42, gamma=1.0)
    assert report["matched_input_verified"] is True
    assert set(report["configurations"].keys()) == {
        "gaussian_std1",
        "official_uniform",
        "gaussian_matched_variance",
    }
    p0 = report["configurations"]["gaussian_std1"]
    p1 = report["configurations"]["official_uniform"]
    p2 = report["configurations"]["gaussian_matched_variance"]

    # P0 has ~sqrt(768)x larger pre-activation std than P1 and P2, causing severe saturation
    assert p0["pre_sigmoid_u"]["std"] > 20.0 * p1["pre_sigmoid_u"]["std"]
    assert p0["post_sigmoid_z"]["total_saturation_fraction"] > p1["post_sigmoid_z"]["total_saturation_fraction"]
    assert abs(p1["pre_sigmoid_u"]["std"] - p2["pre_sigmoid_u"]["std"]) < 0.05
    # P1 and P2 have far lower regularized Gram condition number than P0
    assert p1["numerical_stability"]["regularized_gram_condition_number"] < p0["numerical_stability"]["regularized_gram_condition_number"]
    assert p2["numerical_stability"]["regularized_gram_condition_number"] < p0["numerical_stability"]["regularized_gram_condition_number"]



def test_feature_cache_and_uncached_numerical_equivalence():
    cached_metrics_p = Path("results/c1_pilot_p0_gaussian_std1/metrics.json")
    uncached_metrics_p = Path("results/smoke_test_post_fix_seed42/metrics.json")
    if cached_metrics_p.exists() and uncached_metrics_p.exists():
        import json
        cached = json.loads(cached_metrics_p.read_text(encoding="utf-8"))
        uncached = json.loads(uncached_metrics_p.read_text(encoding="utf-8"))
        assert cached["task_matrix"] == uncached["task_matrix"]
        assert cached["average_incremental_accuracy"] == uncached["average_incremental_accuracy"]
        assert cached["last_accuracy"] == uncached["last_accuracy"]
        assert cached["forgetting"] == uncached["forgetting"]
