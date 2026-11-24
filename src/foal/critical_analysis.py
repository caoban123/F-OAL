"""Theoretical stress-testing utilities and streaming analytic remedies for F-OAL (Lab 2).

Implements the mathematical components derived in `docs/CRITICAL_ANALYSIS_REPORT.md`
without modifying the baseline reproduction modules (`rls.py`, `model.py`, `projection.py`):
1. Exponential long-tail OCIL stream construction and head/medium/tail evaluation splits (L1).
2. `BalancedOnlineRidgeRLS`: one-pass, zero-replay sample-weighted Woodbury RLS and
   inference-time power-norm column calibration (FD1).
3. Deterministic, seeded image corruptions (`gaussian_noise`, `gaussian_blur`,
   `jpeg_compression`, `contrast`) for covariate-shift evaluation (L2a).
4. Spectral Tikhonov diagnostics, pre-projection spherical L2-normalization, and
   closed-form Task-1 Generalized Cross-Validation (GCV) / trace-scaled adaptive ridge (L2b, FD2).
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .rls import OnlineRidgeRLS


# =============================================================================
# 1. Long-Tail OCIL Stream Generation & Group Metrics (Limitation L1)
# =============================================================================

def make_longtail_class_counts(
    num_classes: int,
    n_max: int,
    imbalance_ratio: float,
    *,
    min_samples: int = 1,
    mode: str = "permuted_global",
    classes_per_task: int | None = None,
    quota_seed: int = 42,
) -> list[int]:
    """Compute deterministic per-class training sample quotas under an exponential long-tail law.

    For imbalance ratio ``rho = N_max / N_min >= 1``, the sorted exponential profile is:
        ``q_r = max(min_samples, round(n_max * rho ** (-r / max(K - 1, 1))))``.

    Supporting ``permuted_global`` (default) and ``within_task`` decouples class-frequency
    imbalance from temporal task arrival order so head, medium, and tail classes appear across
    all incremental tasks rather than confounding tail classes with late tasks.

    Args:
        num_classes: Total number of incremental classes ``C``.
        n_max: Maximum samples per head class ``N_max``.
        imbalance_ratio: Ratio ``rho = N_max / N_min`` (``>= 1.0``; ``1.0`` yields balanced).
        min_samples: Floor on the number of samples for the rarest class.
        mode: One of ``'permuted_global'``, ``'within_task'``, or ``'monotonic'``.
        classes_per_task: Required when ``mode == 'within_task'``.
        quota_seed: Seed controlling quota-to-class assignment when ``mode == 'permuted_global'``.

    Returns:
        List of integer sample quotas of length ``num_classes`` indexed by contiguous class ID.
    """
    if num_classes < 1 or n_max < 1 or min_samples < 1:
        raise ValueError("num_classes, n_max, and min_samples must be >= 1")
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1.0")

    def _exp_curve(length: int) -> list[int]:
        if length == 1 or imbalance_ratio == 1.0:
            return [int(max(min_samples, n_max))] * length
        out: list[int] = []
        denom = float(length - 1)
        for r in range(length):
            raw = float(n_max) * (float(imbalance_ratio) ** (-float(r) / denom))
            out.append(int(max(min_samples, int(round(raw)))))
        return out

    if mode == "monotonic":
        return _exp_curve(num_classes)
    if mode == "permuted_global":
        base_quotas = _exp_curve(num_classes)
        if imbalance_ratio == 1.0:
            return base_quotas
        rng = random.Random(int(quota_seed))
        perm = list(range(num_classes))
        rng.shuffle(perm)
        quotas = [0] * num_classes
        for rank_idx, cls_id in enumerate(perm):
            quotas[cls_id] = base_quotas[rank_idx]
        return quotas
    if mode == "within_task":
        if not classes_per_task or classes_per_task < 1 or num_classes % classes_per_task != 0:
            raise ValueError("within_task mode requires classes_per_task dividing num_classes")
        task_curve = _exp_curve(classes_per_task)
        quotas = []
        num_tasks = num_classes // classes_per_task
        for t in range(num_tasks):
            rng = random.Random(int(quota_seed) + t)
            block = list(task_curve)
            if imbalance_ratio > 1.0:
                rng.shuffle(block)
            quotas.extend(block)
        return quotas
    raise ValueError(f"Unsupported long-tail mode: '{mode}'")


def subsample_longtail_indices(
    indices_by_class: Mapping[int, Sequence[int]],
    quotas_by_class: Sequence[int] | Mapping[int, int],
    *,
    seed: int = 42,
) -> tuple[dict[int, list[int]], str]:
    """Deterministically subsample training dataset indices per contiguous class according to quotas.

    Args:
        indices_by_class: Mapping from contiguous class ID ``c`` to full available sample indices.
        quotas_by_class: Sequence or mapping giving desired sample count ``N_c`` for each class ``c``.
        seed: Base random seed for isolated per-class index selection.

    Returns:
        Tuple ``(selected_by_class, sha256_hex)`` of selected index lists and their SHA256 digest.
    """
    selected: dict[int, list[int]] = {}
    hasher = hashlib.sha256()
    for c in sorted(indices_by_class.keys()):
        pool = list(indices_by_class[c])
        quota = int(quotas_by_class[c])
        if quota < 1:
            raise ValueError(f"Quota for class {c} must be >= 1, got {quota}")
        if quota >= len(pool):
            chosen = list(pool)
        else:
            rng = random.Random(int(seed) * 10007 + int(c))
            chosen = rng.sample(pool, quota)
        selected[c] = chosen
        hasher.update(f"{c}:{','.join(map(str, chosen))};".encode("utf-8"))
    return selected, hasher.hexdigest()


def split_head_med_tail_classes(quotas: Sequence[int]) -> dict[str, list[int]]:
    """Partition contiguous class IDs into Head (top 1/3), Medium (middle 1/3), and Tail (bottom 1/3).

    Args:
        quotas: Per-class training sample counts of length ``C``.

    Returns:
        Dictionary with keys ``'head'``, ``'medium'``, and ``'tail'`` listing contiguous class IDs.
    """
    num_classes = len(quotas)
    if num_classes == 0:
        return {"head": [], "medium": [], "tail": []}
    ranked = sorted(range(num_classes), key=lambda c: (-int(quotas[c]), c))
    n_head = max(1, num_classes // 3)
    n_med = max(1, (num_classes - n_head) // 2) if num_classes >= 3 else 0
    return {
        "head": sorted(ranked[:n_head]),
        "medium": sorted(ranked[n_head:n_head + n_med]),
        "tail": sorted(ranked[n_head + n_med:]),
    }


def evaluate_longtail_groups(
    predictions: Sequence[int] | np.ndarray | torch.Tensor,
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    quotas: Sequence[int],
) -> dict[str, Any]:
    """Compute overall, per-class, and Head/Medium/Tail group accuracies plus harmonic mean.

    Args:
        predictions: Predicted contiguous class IDs on the evaluation set, shape ``(N,)``.
        targets: Ground-truth contiguous class IDs on the evaluation set, shape ``(N,)``.
        quotas: Per-class training sample counts of length ``C``.

    Returns:
        Dictionary with ``overall_acc``, ``head_acc``, ``medium_acc``, ``tail_acc``,
        ``harmonic_mean``, and ``per_class_acc`` (all in ``[0, 1]``).
    """
    preds = np.asarray(predictions, dtype=np.int64).reshape(-1)
    trues = np.asarray(targets, dtype=np.int64).reshape(-1)
    if preds.shape != trues.shape or preds.size == 0:
        raise ValueError("predictions and targets must be non-empty 1D arrays of identical length")

    num_classes = len(quotas)
    per_class_acc: list[float] = []
    for c in range(num_classes):
        mask = trues == c
        per_class_acc.append(float((preds[mask] == c).mean()) if np.any(mask) else float("nan"))

    groups = split_head_med_tail_classes(quotas)

    def _mean_group(cls_ids: list[int]) -> float:
        vals = [per_class_acc[c] for c in cls_ids if not math.isnan(per_class_acc[c])]
        return float(np.mean(vals)) if vals else float("nan")

    head_acc = _mean_group(groups["head"])
    med_acc = _mean_group(groups["medium"])
    tail_acc = _mean_group(groups["tail"])
    valid_groups = [v for v in (head_acc, med_acc, tail_acc) if not math.isnan(v)]
    if valid_groups and all(v > 0.0 for v in valid_groups):
        hmean = float(len(valid_groups) / sum(1.0 / v for v in valid_groups))
    else:
        hmean = 0.0

    return {
        "overall_acc": float((preds == trues).mean()),
        "head_acc": head_acc,
        "medium_acc": med_acc,
        "tail_acc": tail_acc,
        "harmonic_mean": hmean,
        "per_class_acc": per_class_acc,
    }


# =============================================================================
# 2. BalancedOnlineRidgeRLS — Zero-Replay Remedy for Long-Tail OCIL (FD1)
# =============================================================================

class BalancedOnlineRidgeRLS(OnlineRidgeRLS):
    """Streaming sample-weighted Woodbury RLS with inference-time norm calibration (FD1).

    Optimizes the weighted ridge objective without storing historical samples:
        ``min_W sum_i omega_i ||y_i - phi_i W||_2^2 + gamma ||W||_F^2``
    via exact row-rescaled Woodbury updates ``(sqrt(omega_i) phi_i, sqrt(omega_i) y_i)``,
    and optionally applies zero-cost column power-norm calibration at inference:
        ``w_tilde_c = w_c / (||w_c||_2 ** norm_tau + eps)``.
    """

    def __init__(
        self,
        feature_dim: int,
        gamma: float = 1.0,
        *,
        weight_mode: str = "none",
        weight_power: float = 0.0,
        ref_count: float = 100.0,
        norm_tau: float = 0.0,
        eps: float = 1e-8,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the balanced online analytic classifier.

        Args:
            feature_dim: Input activation dimension ``D``.
            gamma: Ridge regularization coefficient ``gamma > 0``.
            weight_mode: ``'none'`` (unweighted), ``'online_running'`` (strictly causal running
                class count ``(ref_count / N_hat_c) ** weight_power``), or ``'provided'``
                (uses explicit ``sample_weights`` passed to ``update``).
            weight_power: Exponent ``beta >= 0`` for inverse-frequency sample weighting.
            ref_count: Reference sample count ``N_ref > 0`` scaling ``online_running`` weights.
            norm_tau: Inference-time column norm exponent ``tau >= 0`` (``0.0`` disables).
            eps: Numerical floor preventing division by zero during norm calibration.
            device: Torch compute device.
            dtype: Floating-point precision (default ``torch.float64``).
        """
        super().__init__(feature_dim=feature_dim, gamma=gamma, device=device, dtype=dtype)
        if weight_mode not in ("none", "online_running", "provided"):
            raise ValueError(f"Unsupported weight_mode: '{weight_mode}'")
        if weight_power < 0.0 or ref_count <= 0.0 or norm_tau < 0.0 or eps <= 0.0:
            raise ValueError("weight_power >= 0, ref_count > 0, norm_tau >= 0, eps > 0 required")
        self.weight_mode = str(weight_mode)
        self.weight_power = float(weight_power)
        self.ref_count = float(ref_count)
        self.norm_tau = float(norm_tau)
        self.eps = float(eps)
        self.class_counts = torch.zeros((0,), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def expand_classes(self, total_classes: int) -> None:
        """Expand classifier columns and online class counters monotonically."""
        prev = self.num_classes
        super().expand_classes(total_classes)
        added = total_classes - prev
        if added > 0:
            zeros = torch.zeros((added,), device=self.device, dtype=torch.long)
            self.class_counts = torch.cat((self.class_counts, zeros), dim=0)

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Perform one exact sample-weighted Woodbury update on a mini-batch.

        Args:
            features: Activation batch of shape ``(B, D)``.
            labels: Integer class labels of shape ``(B,)``.
            sample_weights: Optional positive weights of shape ``(B,)`` when
                ``weight_mode == 'provided'`` (or explicitly supplied).
        """
        H = features.to(device=self.device, dtype=self.dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        if H.ndim != 2 or H.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features [batch, {self.feature_dim}]")
        if y.ndim != 1 or y.shape[0] != H.shape[0] or H.shape[0] == 0:
            raise ValueError("Expected non-empty labels [batch]")
        if self.num_classes == 0 or y.min() < 0 or y.max() >= self.num_classes:
            raise ValueError("Call expand_classes before updating with new labels")
        if not torch.all(torch.isfinite(H)):
            raise ValueError("Input features contain NaN or Inf values")

        batch_counts = torch.bincount(y, minlength=self.num_classes)
        self.class_counts = self.class_counts + batch_counts

        if sample_weights is not None:
            w = sample_weights.to(device=self.device, dtype=self.dtype)
            if w.shape != (H.shape[0],) or torch.any(w <= 0) or not torch.all(torch.isfinite(w)):
                raise ValueError("sample_weights must be finite, strictly positive, shape (B,)")
        elif self.weight_mode == "online_running" and self.weight_power > 0.0:
            counts_for_batch = self.class_counts[y].to(dtype=self.dtype).clamp_min(1.0)
            w = (self.ref_count / counts_for_batch) ** self.weight_power
        elif self.weight_mode == "provided":
            raise ValueError("sample_weights must be supplied when weight_mode == 'provided'")
        else:
            w = None

        Y = F.one_hot(y, num_classes=self.num_classes).to(self.dtype)
        if w is not None:
            sqrt_w = torch.sqrt(w).unsqueeze(1)
            H_eff = H * sqrt_w
            Y_eff = Y * sqrt_w
        else:
            H_eff = H
            Y_eff = Y

        K = self.R @ H_eff.T
        A = torch.eye(H_eff.shape[0], device=self.device, dtype=self.dtype) + H_eff @ K
        gain = torch.linalg.solve(A, K.T).T
        residual = Y_eff - H_eff @ self.W
        self.W = self.W + gain @ residual
        self.R = self.R - gain @ K.T
        self.R = 0.5 * (self.R + self.R.T)

        if not (torch.all(torch.isfinite(self.W)) and torch.all(torch.isfinite(self.R))):
            raise RuntimeError("BalancedOnlineRidgeRLS update produced NaN or Inf")
        self.samples_seen += H.shape[0]

    @torch.no_grad()
    def column_norms(self) -> torch.Tensor:
        """Return Euclidean L2 norm ``||w_c||_2`` of each registered class column.

        Returns:
            1D tensor of shape ``(num_classes,)`` on CPU in ``float64``.
        """
        return torch.linalg.vector_norm(self.W, ord=2, dim=0).detach().cpu().to(torch.float64)

    @torch.no_grad()
    def effective_weights(self) -> torch.Tensor:
        """Return inference-time weight matrix after optional power-norm calibration."""
        if self.norm_tau == 0.0:
            return self.W
        norms = torch.linalg.vector_norm(self.W, ord=2, dim=0, keepdim=True).clamp_min(self.eps)
        return self.W / (norms ** self.norm_tau)

    @torch.no_grad()
    def scores(self, features: torch.Tensor) -> torch.Tensor:
        """Compute classification scores using calibrated or raw analytic weights.

        Args:
            features: Activation matrix of shape ``(B, D)``.

        Returns:
            Score tensor of shape ``(B, num_classes)``.
        """
        if self.num_classes == 0:
            raise RuntimeError("No classes have been seen yet")
        H = features.to(device=self.device, dtype=self.dtype)
        return H @ self.effective_weights()


# =============================================================================
# 3. Deterministic Image Corruptions for Covariate-Shift Stress Tests (L2a)
# =============================================================================

GAUSSIAN_NOISE_STD_BY_SEVERITY = {1: 0.04, 2: 0.08, 3: 0.12, 4: 0.18, 5: 0.26}
GAUSSIAN_BLUR_PARAMS_BY_SEVERITY = {
    1: (3, 0.6),
    2: (5, 1.0),
    3: (7, 1.6),
    4: (9, 2.4),
    5: (13, 3.5),
}
JPEG_QUALITY_BY_SEVERITY = {1: 65, 2: 45, 3: 30, 4: 18, 5: 10}
CONTRAST_ALPHA_BY_SEVERITY = {1: 0.25, 2: 0.45, 3: 0.65, 4: 0.80, 5: 0.90}


def _gaussian_kernel_2d(kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = torch.outer(g1d, g1d)
    return g2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


@torch.no_grad()
def apply_image_corruption(
    images: torch.Tensor,
    corruption: str,
    severity: int,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Apply a deterministic sensor/environmental corruption to `[0, 1]` image tensors.

    Supports both single images ``(C, H, W)`` and batches ``(B, C, H, W)`` assumed to lie
    in ``[0, 1]`` prior to any channel mean/std normalization.

    Args:
        images: Float tensor in ``[0, 1]`` of shape ``(C, H, W)`` or ``(B, C, H, W)``.
        corruption: One of ``'clean'``, ``'gaussian_noise'``, ``'gaussian_blur'``,
            ``'jpeg_compression'``, or ``'contrast'``.
        severity: Integer severity in ``{0, 1, 2, 3, 4, 5}`` (``0`` always returns clean copy).
        seed: Deterministic RNG seed for stochastic corruptions.

    Returns:
        Corrupted float tensor in ``[0, 1]`` with the exact same shape, device, and dtype.
    """
    if severity < 0 or severity > 5:
        raise ValueError(f"severity must be in 0..5, got {severity}")
    valid_corruptions = ("clean", "gaussian_noise", "gaussian_blur", "jpeg_compression", "contrast")
    if corruption not in valid_corruptions:
        raise ValueError(f"Unsupported corruption '{corruption}'. Expected one of {valid_corruptions}")
    if severity == 0 or corruption == "clean":
        return images.clone()

    squeeze_back = False
    x = images
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze_back = True
    elif x.ndim != 4:
        raise ValueError("images must have shape (C, H, W) or (B, C, H, W)")

    if corruption == "gaussian_noise":
        std = GAUSSIAN_NOISE_STD_BY_SEVERITY[severity]
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(device=x.device, dtype=x.dtype)
        out = (x + noise * std).clamp(0.0, 1.0)
    elif corruption == "gaussian_blur":
        ksize, sigma = GAUSSIAN_BLUR_PARAMS_BY_SEVERITY[severity]
        pad = ksize // 2
        kernel = _gaussian_kernel_2d(ksize, sigma, x.shape[1], x.device, x.dtype)
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        out = F.conv2d(padded, kernel, groups=x.shape[1]).clamp(0.0, 1.0)
    elif corruption == "contrast":
        alpha = CONTRAST_ALPHA_BY_SEVERITY[severity]
        mean = x.mean(dim=(-2, -1), keepdim=True)
        out = ((1.0 - alpha) * x + alpha * mean).clamp(0.0, 1.0)
    elif corruption == "jpeg_compression":
        quality = JPEG_QUALITY_BY_SEVERITY[severity]
        x_cpu = (x.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        decoded = []
        for idx in range(x_cpu.shape[0]):
            hwc = np.transpose(x_cpu[idx], (1, 2, 0))
            pil_img = Image.fromarray(hwc, mode="RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            rec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
            decoded.append(np.transpose(rec, (2, 0, 1)))
        out = torch.from_numpy(np.stack(decoded, axis=0)).to(device=x.device, dtype=x.dtype)

    return out.squeeze(0) if squeeze_back else out


# =============================================================================
# 4. Spectral Diagnostics, Pre-L2 Norm & Adaptive Ridge / GCV (L2b & FD2)
# =============================================================================

@torch.no_grad()
def pre_l2_normalize(features: torch.Tensor, *, scale_sqrt_dim: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Project backbone features onto the sphere of radius ``sqrt(d_in)`` (or unit sphere).

    Stabilizes pre-projection activation variance ``Var(h @ P)`` under input covariate shift.

    Args:
        features: Backbone feature tensor of shape ``(..., d_in)``.
        scale_sqrt_dim: If True, multiplies unit-norm vectors by ``sqrt(d_in)`` to match LayerNorm scale.
        eps: Numerical floor for zero-norm protection.

    Returns:
        Normalized tensor of identical shape and dtype.
    """
    d_in = features.shape[-1]
    norm = torch.linalg.vector_norm(features, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    unit = features / norm
    return unit * math.sqrt(float(d_in)) if scale_sqrt_dim else unit


@torch.no_grad()
def compute_spectral_diagnostics(features: torch.Tensor, gamma: float = 1.0) -> dict[str, float]:
    """Compute exact spectral conditioning metrics of ``Phi.T @ Phi + gamma * I_D``.

    Uses the smaller of ``Phi.T @ Phi`` ``(D x D)`` and ``Phi @ Phi.T`` ``(V x V)`` so that
    evaluating high-dimensional projections (e.g., ``D = 12,000`` on ``V = 1,600`` DTD samples)
    runs in ``O(min(V, D)^3)`` rather than ``O(D^3)``.

    Args:
        features: Activation matrix ``Phi`` of shape ``(V, D)``.
        gamma: Ridge regularization parameter ``gamma > 0``.

    Returns:
        Dictionary with ``num_samples``, ``feature_dim``, ``trace``, ``max_eig``, ``min_eig``,
        ``regularized_cond``, ``effective_rank`` (Tikhonov degrees of freedom), and ``max_filter_gain``.
    """
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D tensor (V, D)")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")

    Phi = features.detach().to(dtype=torch.float64, device="cpu")
    v_cnt, d_dim = Phi.shape
    if v_cnt < d_dim:
        gram_small = Phi @ Phi.T
    else:
        gram_small = Phi.T @ Phi
    gram_small = 0.5 * (gram_small + gram_small.T)
    pos_eigs = torch.linalg.eigvalsh(gram_small).clamp_min(0.0)

    max_eig = float(pos_eigs[-1].item())
    min_eig = float(pos_eigs[0].item()) if v_cnt >= d_dim else 0.0
    trace_val = float(pos_eigs.sum().item())
    reg_cond = float((max_eig + gamma) / (min_eig + gamma))
    eff_rank = float((pos_eigs / (pos_eigs + gamma)).sum().item())
    sigmas = torch.sqrt(pos_eigs)
    max_gain = float((sigmas / (pos_eigs + gamma)).max().item())

    return {
        "num_samples": float(v_cnt),
        "feature_dim": float(d_dim),
        "trace": trace_val,
        "max_eig": max_eig,
        "min_eig": min_eig,
        "regularized_cond": reg_cond,
        "effective_rank": eff_rank,
        "max_filter_gain": max_gain,
    }


@torch.no_grad()
def compute_trace_scaled_gamma(
    features_task1: torch.Tensor,
    *,
    gamma_0: float = 1.0,
    total_expected_samples: int | None = None,
    ref_dim: int = 1000,
) -> float:
    """Compute trace-and-dimension scaled ridge parameter ``gamma_auto`` from Task-1 activations (FD2).

    Formula:
        ``gamma_auto = gamma_0 * (tr(Phi_1.T @ Phi_1) / (V_1 * (ref_dim * 0.25))) * max(1.0, D / V_est)``
    Normalized so that at ``D = 1000``, ``V_est >= 1000``, and standard sigmoid energy
    ``E[||phi||^2] ~ 0.25 * D``, ``gamma_auto`` stays near ``gamma_0``, while automatically scaling
    up when ``D`` grows or ``D > V_est`` enters the rank-deficient interpolation regime.

    Args:
        features_task1: Task-1 activation matrix ``Phi_1`` of shape ``(V_1, D)``.
        gamma_0: Base dimensionless regularization constant (default ``1.0``).
        total_expected_samples: Estimated total samples ``V_est`` across stream (defaults to ``10 * V_1``).
        ref_dim: Reference projection dimension where ``gamma_0`` is calibrated (default ``1000``).

    Returns:
        Positive scalar ``gamma_auto``.
    """
    if features_task1.ndim != 2 or features_task1.shape[0] == 0:
        raise ValueError("features_task1 must be non-empty 2D tensor")
    if gamma_0 <= 0.0 or ref_dim < 1:
        raise ValueError("gamma_0 > 0 and ref_dim >= 1 required")

    Phi1 = features_task1.detach().to(dtype=torch.float64)
    v1, d_dim = Phi1.shape
    v_est = float(total_expected_samples if total_expected_samples is not None else 10 * v1)
    mean_sq_norm = float((Phi1 * Phi1).sum().item()) / float(v1)
    ref_energy = 0.25 * float(ref_dim)
    overparam_factor = max(1.0, float(d_dim) / max(v_est, 1.0))
    return float(max(1e-6, gamma_0 * (mean_sq_norm / ref_energy) * overparam_factor))


@torch.no_grad()
def select_gcv_gamma(
    features_task1: torch.Tensor,
    labels_task1: torch.Tensor,
    num_classes_task1: int,
    *,
    candidate_gammas: Sequence[float] | None = None,
    horizon_tasks: float = 1.0,
) -> dict[str, Any]:
    """Select optimal ridge regularization ``gamma*`` on Task 1 via closed-form Generalized Cross-Validation.

    Uses a single thin SVD of ``Phi_1`` to evaluate ``GCV(gamma)`` across all candidates in
    ``O(V_1 * D * min(V_1, D))`` without requiring a held-out validation split.

    Args:
        features_task1: Task-1 activation matrix ``Phi_1`` of shape ``(V_1, D)``.
        labels_task1: Task-1 integer class labels in ``0..num_classes_task1-1``, shape ``(V_1,)``.
        num_classes_task1: Number of classes in Task 1.
        candidate_gammas: Optional sequence of positive candidate ``gamma`` values.
        horizon_tasks: Multiplier scaling ``gamma*_task1`` to full stream horizon (default ``1.0``).

    Returns:
        Dictionary with ``best_gamma_task1``, ``best_gamma_stream``, ``best_gcv_score``, and ``scores``.
    """
    Phi = features_task1.detach().to(dtype=torch.float64, device="cpu")
    y = labels_task1.detach().to(dtype=torch.long, device="cpu")
    v1, _ = Phi.shape
    if v1 < 2:
        raise ValueError("select_gcv_gamma requires at least 2 samples in Task 1")
    Y = F.one_hot(y, num_classes=int(num_classes_task1)).to(dtype=torch.float64)

    if candidate_gammas is None:
        candidate_gammas = [float(10.0 ** p) for p in np.linspace(-3.0, 3.0, 25)]

    U, S, _ = torch.linalg.svd(Phi, full_matrices=False)
    s2 = S * S
    UTY = U.T @ Y
    uty_sq = (UTY * UTY).sum(dim=1)
    total_y_sq = float((Y * Y).sum().item())
    perp_y_sq = max(0.0, total_y_sq - float(uty_sq.sum().item()))

    scores: dict[float, float] = {}
    best_gamma = float(candidate_gammas[0])
    best_score = float("inf")

    for g in candidate_gammas:
        g_val = float(g)
        if g_val <= 0.0:
            continue
        shrink = g_val / (s2 + g_val)
        resid_sq = perp_y_sq + float(((shrink * shrink) * uty_sq).sum().item())
        tr_hat = float((s2 / (s2 + g_val)).sum().item())
        denom = max(1e-8, 1.0 - tr_hat / float(v1)) ** 2
        gcv = (resid_sq / float(v1)) / denom
        scores[g_val] = float(gcv)
        if gcv < best_score:
            best_score = float(gcv)
            best_gamma = g_val

    return {
        "best_gamma_task1": best_gamma,
        "best_gamma_stream": float(best_gamma * max(1.0, float(horizon_tasks))),
        "best_gcv_score": best_score,
        "scores": scores,
    }
