"""CPU-safe unit tests for Lab 2 Critical Analysis & Empirical Demonstration utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.foal.critical_analysis import (
    BalancedOnlineRidgeRLS,
    apply_image_corruption,
    compute_spectral_diagnostics,
    compute_trace_scaled_gamma,
    evaluate_longtail_groups,
    make_longtail_class_counts,
    pre_l2_normalize,
    select_gcv_gamma,
    split_head_med_tail_classes,
    subsample_longtail_indices,
)
from src.foal.rls import OnlineRidgeRLS


def test_longtail_quotas_and_deterministic_subsampling() -> None:
    """Verify exponential long-tail quotas, modes, and SHA256 reproducibility."""
    balanced = make_longtail_class_counts(100, 500, 1.0, mode="permuted_global", quota_seed=42)
    assert balanced == [500] * 100

    lt_global = make_longtail_class_counts(100, 500, 50.0, mode="permuted_global", quota_seed=42)
    assert len(lt_global) == 100
    assert max(lt_global) == 500
    assert min(lt_global) == 10
    # Permuted global ensures first 10 classes are not all 500-sample head classes
    assert len(set(lt_global[:10])) > 3

    lt_within = make_longtail_class_counts(
        40, 40, 20.0, mode="within_task", classes_per_task=4, quota_seed=42
    )
    assert len(lt_within) == 40
    for task_idx in range(10):
        block = lt_within[task_idx * 4:(task_idx + 1) * 4]
        assert max(block) == 40
        assert min(block) == 2

    pool = {c: list(range(c * 50, (c + 1) * 50)) for c in range(12)}
    quotas = make_longtail_class_counts(12, 40, 10.0, mode="monotonic")
    sel1, sha1 = subsample_longtail_indices(pool, quotas, seed=123)
    sel2, sha2 = subsample_longtail_indices(pool, quotas, seed=123)
    _, sha3 = subsample_longtail_indices(pool, quotas, seed=2026)
    assert sel1 == sel2
    assert sha1 == sha2
    assert sha1 != sha3
    assert [len(sel1[c]) for c in range(12)] == quotas


def test_balanced_rls_matches_closed_form_weighted_ridge_and_baseline() -> None:
    """Verify exact equivalence between streaming BalancedOnlineRidgeRLS and closed-form weighted ridge."""
    gen = torch.Generator().manual_seed(42)
    n_samples, dim, n_classes = 96, 24, 6
    gamma = 1.5
    H = torch.randn(n_samples, dim, generator=gen, dtype=torch.float64)
    y = torch.randint(0, n_classes, (n_samples,), generator=gen, dtype=torch.long)
    weights = torch.rand(n_samples, generator=gen, dtype=torch.float64) * 4.0 + 0.25

    # 1. Closed-form weighted ridge: (H.T @ diag(w) @ H + gamma * I)^(-1) @ H.T @ diag(w) @ Y
    Y = F.one_hot(y, num_classes=n_classes).to(torch.float64)
    W_diag = torch.diag(weights)
    A_closed = H.T @ W_diag @ H + gamma * torch.eye(dim, dtype=torch.float64)
    W_closed = torch.linalg.solve(A_closed, H.T @ W_diag @ Y)

    # 2. Streaming mini-batch Weighted Woodbury with expanding classes
    clf = BalancedOnlineRidgeRLS(dim, gamma=gamma, weight_mode="provided", dtype=torch.float64)
    clf.expand_classes(3)
    mask_first = y < 3
    H1, y1, w1 = H[mask_first], y[mask_first], weights[mask_first]
    for start in range(0, len(y1), 7):
        clf.update(H1[start:start + 7], y1[start:start + 7], sample_weights=w1[start:start + 7])

    clf.expand_classes(n_classes)
    H2, y2, w2 = H[~mask_first], y[~mask_first], weights[~mask_first]
    for start in range(0, len(y2), 11):
        clf.update(H2[start:start + 11], y2[start:start + 11], sample_weights=w2[start:start + 11])

    max_diff = float((clf.W - W_closed).abs().max().item())
    assert max_diff < 1e-11, f"Weighted Woodbury diverged from closed-form: {max_diff}"

    # 3. Unweighted mode matches baseline OnlineRidgeRLS bitwise/numerically
    base = OnlineRidgeRLS(dim, gamma=gamma, dtype=torch.float64)
    unw = BalancedOnlineRidgeRLS(dim, gamma=gamma, weight_mode="none", norm_tau=0.0, dtype=torch.float64)
    base.expand_classes(n_classes)
    unw.expand_classes(n_classes)
    base.update(H, y)
    unw.update(H, y)
    assert torch.allclose(base.W, unw.W, atol=1e-13)


def test_longtail_collapse_on_synthetic_stream_and_recovery_by_fd1() -> None:
    """Demonstrate Theorem L1.2 minority weight-norm collapse and FD1 recovery on synthetic data."""
    gen = torch.Generator().manual_seed(7)
    dim, n_classes = 32, 6
    quotas = [120, 100, 50, 20, 5, 3]
    centers = F.normalize(torch.randn(n_classes, dim, generator=gen, dtype=torch.float64), dim=1) * 2.0
    # Add positive offset mimicking sigmoid activations in (0, 1)^D
    offset = torch.full((1, dim), 0.5, dtype=torch.float64)

    train_feats, train_labels = [], []
    test_feats, test_labels = [], []
    for c, q in enumerate(quotas):
        tr = torch.sigmoid(centers[c:c + 1] + 0.35 * torch.randn(q, dim, generator=gen, dtype=torch.float64) + offset)
        te = torch.sigmoid(centers[c:c + 1] + 0.35 * torch.randn(60, dim, generator=gen, dtype=torch.float64) + offset)
        train_feats.append(tr)
        train_labels.append(torch.full((q,), c, dtype=torch.long))
        test_feats.append(te)
        test_labels.append(torch.full((60,), c, dtype=torch.long))

    H_tr = torch.cat(train_feats, dim=0)
    y_tr = torch.cat(train_labels, dim=0)
    H_te = torch.cat(test_feats, dim=0)
    y_te = torch.cat(test_labels, dim=0)

    vanilla = BalancedOnlineRidgeRLS(dim, gamma=1.0, weight_mode="none", norm_tau=0.0)
    vanilla.expand_classes(n_classes)
    vanilla.update(H_tr, y_tr)
    norms_vanilla = vanilla.column_norms()
    # Head class norm strictly exceeds tail class norm under unweighted one-hot MSE
    assert norms_vanilla[0] > norms_vanilla[-1] * 2.0

    res_van = evaluate_longtail_groups(vanilla.predict(H_te).numpy(), y_te.numpy(), quotas)

    remedy = BalancedOnlineRidgeRLS(
        dim, gamma=1.0, weight_mode="online_running", weight_power=0.75, ref_count=60.0, norm_tau=0.75
    )
    remedy.expand_classes(n_classes)
    # Shuffle stream to test true online causality
    perm = torch.randperm(len(y_tr), generator=gen)
    for start in range(0, len(y_tr), 16):
        idx = perm[start:start + 16]
        remedy.update(H_tr[idx], y_tr[idx])
    res_rem = evaluate_longtail_groups(remedy.predict(H_te).numpy(), y_te.numpy(), quotas)

    assert res_rem["tail_acc"] > res_van["tail_acc"]
    assert split_head_med_tail_classes(quotas)["tail"] == [4, 5]


def test_deterministic_corruptions_and_spectral_gcv_diagnostics() -> None:
    """Verify deterministic image corruptions, fast dual-Gram spectral diagnostics, and GCV gamma."""
    img = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(99))
    for cname in ("gaussian_noise", "gaussian_blur", "jpeg_compression", "contrast"):
        c0 = apply_image_corruption(img, cname, 0, seed=42)
        assert torch.equal(c0, img)
        c3_a = apply_image_corruption(img, cname, 3, seed=42)
        c3_b = apply_image_corruption(img, cname, 3, seed=42)
        assert torch.allclose(c3_a, c3_b)
        assert c3_a.min() >= 0.0 and c3_a.max() <= 1.0
        assert not torch.allclose(c3_a, img)

    # Rank-deficient regime V < D: min_eig must be 0.0 and dual Gram matches full Gram trace
    Phi = torch.sigmoid(torch.randn(40, 120, generator=torch.Generator().manual_seed(11)))
    spec = compute_spectral_diagnostics(Phi, gamma=0.01)
    assert spec["min_eig"] == 0.0
    assert spec["effective_rank"] <= 40.0
    assert abs(spec["trace"] - float((Phi.double() ** 2).sum().item())) < 1e-7

    normed = pre_l2_normalize(Phi, scale_sqrt_dim=True)
    row_norms = torch.linalg.vector_norm(normed, dim=-1)
    assert torch.allclose(row_norms, torch.full_like(row_norms, (120.0) ** 0.5), atol=1e-5)

    g_auto = compute_trace_scaled_gamma(Phi, gamma_0=1.0, total_expected_samples=80, ref_dim=100)
    assert g_auto > 0.0

    y1 = torch.arange(40, dtype=torch.long) % 4
    gcv = select_gcv_gamma(Phi, y1, num_classes_task1=4)
    assert gcv["best_gamma_task1"] > 0.0
    assert len(gcv["scores"]) == 25
