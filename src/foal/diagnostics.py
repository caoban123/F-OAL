"""Matched-input Smooth Projection diagnostics for Experiment C1 (Section 9).

Evaluates `P0` (`gaussian_std1`), `P1` (`official_uniform`), and `P2` (`gaussian_matched_variance`)
on the exact same frozen ViT feature matrix h in R^{N x d} extracted from a deterministic,
multi-class subset of the CIFAR-100 training split (strictly without touching test labels).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np
import torch

from .projection import SmoothProjection, VALID_PROJECTION_INITIALIZATIONS

QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@torch.no_grad()
def compute_projection_diagnostics(
    features: torch.Tensor,
    projection: SmoothProjection,
    gamma: float = 1.0,
    histogram_bins: int = 20,
) -> dict[str, Any]:
    """Compute pre-sigmoid, post-sigmoid, saturation, and Gram conditioning diagnostics.

    Parameters
    ----------
    features:
        2D tensor of shape [N, d] containing frozen backbone features h.
    projection:
        Initialized `SmoothProjection` module with frozen weight matrix P of shape [d, D].
    gamma:
        Ridge regularization coefficient gamma > 0 for Gram conditioning analysis.
    histogram_bins:
        Number of bins for pre-sigmoid and post-sigmoid empirical histograms.
    """
    if features.ndim != 2:
        raise ValueError(f'Expected 2D feature tensor [N, d], got shape {tuple(features.shape)}')
    n_samples, feat_dim = features.shape
    if feat_dim != projection.input_dim:
        raise ValueError(
            f'Feature dimension {feat_dim} does not match projection input_dim {projection.input_dim}'
        )

    h_cpu = features.detach().cpu().float().contiguous()
    w_cpu = projection.weight.detach().cpu().float().contiguous()

    # Compute pre-sigmoid u = h @ P and post-sigmoid z = sigmoid(u) in float64 for high-precision stats
    u = (h_cpu @ w_cpu).double()
    z = torch.sigmoid(u)

    u_flat = u.reshape(-1)
    z_flat = z.reshape(-1)

    q_tensor = torch.tensor(QUANTILE_LEVELS, dtype=torch.float64)
    u_quantiles_vals = torch.quantile(u_flat, q_tensor).tolist()
    z_quantiles_vals = torch.quantile(z_flat, q_tensor).tolist()
    u_quantiles = {f'q{int(q * 100):02d}': float(val) for q, val in zip(QUANTILE_LEVELS, u_quantiles_vals)}
    z_quantiles = {f'q{int(q * 100):02d}': float(val) for q, val in zip(QUANTILE_LEVELS, z_quantiles_vals)}

    u_counts, u_edges = np.histogram(u_flat.numpy(), bins=histogram_bins)
    z_counts, z_edges = np.histogram(z_flat.numpy(), bins=histogram_bins, range=(0.0, 1.0))
    total_elements = float(u_flat.numel())

    # Saturation fractions
    frac_low = float((z < 0.01).double().mean().item())
    frac_high = float((z > 0.99).double().mean().item())
    frac_total_sat = float(((z < 0.01) | (z > 0.99)).double().mean().item())

    # Per-feature variance / std across N samples
    if n_samples > 1:
        col_std = z.std(dim=0, unbiased=True)
        col_var = z.var(dim=0, unbiased=True)
    else:
        col_std = torch.zeros(projection.output_dim, dtype=torch.float64)
        col_var = torch.zeros(projection.output_dim, dtype=torch.float64)

    near_const_1e3 = int((col_std < 1e-3).sum().item())
    near_const_1e2 = int((col_std < 1e-2).sum().item())
    near_zero_var_1e6 = int((col_var < 1e-6).sum().item())

    # Gram matrix G = Z^T Z (D x D symmetric PSD) and G_gamma = Z^T Z + gamma * I_D
    gram = z.T @ z
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    eig_max = float(eigvals_clamped[-1].item())
    eig_min = float(eigvals_clamped[0].item())

    reg_eigvals = eigvals_clamped + float(gamma)
    reg_cond = float((reg_eigvals[-1] / reg_eigvals[0]).item())

    # Effective rank (Roy & Vetterli, 2007): exp(Shannon entropy of normalized eigenvalues)
    eig_sum = float(eigvals_clamped.sum().item())
    if eig_sum > 0.0:
        p = eigvals_clamped / eig_sum
        p_pos = p[p > 1e-15]
        shannon_h = float(-(p_pos * torch.log(p_pos)).sum().item())
        erank_unreg = float(math.exp(shannon_h))
    else:
        erank_unreg = 0.0

    reg_sum = float(reg_eigvals.sum().item())
    q_dist = reg_eigvals / reg_sum
    erank_reg = float(math.exp(float(-(q_dist * torch.log(q_dist)).sum().item())))

    num_rank_1e6 = int((eigvals_clamped > (1e-6 * max(eig_max, 1e-12))).sum().item())

    return {
        'projection_metadata': projection.get_metadata(),
        'input_features': {
            'num_samples': int(n_samples),
            'feature_dim': int(feat_dim),
            'mean': float(h_cpu.double().mean().item()),
            'std': float(h_cpu.double().std(unbiased=True).item()),
            'l2_norm_mean': float(h_cpu.double().norm(dim=1).mean().item()),
            'sha256': hashlib.sha256(h_cpu.numpy().tobytes()).hexdigest(),
        },
        'pre_sigmoid_u': {
            'mean': float(u.mean().item()),
            'std': float(u.std(unbiased=True).item()),
            'min': float(u.min().item()),
            'max': float(u.max().item()),
            'quantiles': u_quantiles,
            'histogram': {
                'bin_edges': [float(e) for e in u_edges.tolist()],
                'counts': [int(c) for c in u_counts.tolist()],
                'fractions': [float(c / total_elements) for c in u_counts.tolist()],
            },
        },
        'post_sigmoid_z': {
            'mean': float(z.mean().item()),
            'std': float(z.std(unbiased=True).item()),
            'min': float(z.min().item()),
            'max': float(z.max().item()),
            'quantiles': z_quantiles,
            'fraction_z_lt_0_01': frac_low,
            'fraction_z_gt_0_99': frac_high,
            'total_saturation_fraction': frac_total_sat,
            'feature_std_min': float(col_std.min().item()),
            'feature_std_median': float(col_std.median().item()),
            'feature_std_max': float(col_std.max().item()),
            'near_constant_features_std_lt_1e3_count': near_const_1e3,
            'near_constant_features_std_lt_1e3_fraction': float(near_const_1e3 / projection.output_dim),
            'near_constant_features_std_lt_1e2_count': near_const_1e2,
            'near_constant_features_std_lt_1e2_fraction': float(near_const_1e2 / projection.output_dim),
            'histogram': {
                'bin_edges': [float(e) for e in z_edges.tolist()],
                'counts': [int(c) for c in z_counts.tolist()],
                'fractions': [float(c / total_elements) for c in z_counts.tolist()],
            },
        },
        'numerical_stability': {
            'gamma': float(gamma),
            'regularized_gram_condition_number': reg_cond,
            'unregularized_gram_max_eigval': eig_max,
            'unregularized_gram_min_eigval': eig_min,
            'effective_rank_unregularized': erank_unreg,
            'effective_rank_regularized': erank_reg,
            'numerical_rank_rel_tol_1e6': num_rank_1e6,
            'near_zero_variance_features_var_lt_1e6_count': near_zero_var_1e6,
        },
    }


@torch.no_grad()
def compare_c1_projections_on_features(
    features: torch.Tensor,
    projection_dim: int = 1000,
    projection_seed: int = 42,
    gamma: float = 1.0,
    initializations: Sequence[str] = VALID_PROJECTION_INITIALIZATIONS,
) -> dict[str, Any]:
    """Evaluate all C1 SmoothProjection initializers on the exact same feature tensor h."""
    input_dim = int(features.shape[1])
    features_sha = hashlib.sha256(features.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()
    results: dict[str, Any] = {}

    for init_name in initializations:
        proj = SmoothProjection(
            input_dim=input_dim,
            output_dim=projection_dim,
            seed=projection_seed,
            initialization=init_name,
        )
        diag = compute_projection_diagnostics(features, proj, gamma=gamma)
        if diag['input_features']['sha256'] != features_sha:
            raise AssertionError('Matched-input invariant violated: feature tensor SHA256 changed!')
        results[init_name] = diag

    return {
        'matched_input_verified': True,
        'input_features_sha256': features_sha,
        'num_samples': int(features.shape[0]),
        'input_dim': input_dim,
        'projection_dim': int(projection_dim),
        'projection_seed': int(projection_seed),
        'gamma': float(gamma),
        'configurations': results,
    }
