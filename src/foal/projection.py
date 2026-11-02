"""Fixed random projection followed by sigmoid (F-OAL Eq. 1) with configurable C1 initializers.

Supported initialization strategies (Experiment C1):
1. `gaussian_std1` (P0 — Current Gaussian Baseline):
       P_{ij} ~ N(0, 1)  (or N(0, gaussian_std^2) when gaussian_std != 1.0)
       Population variance = 1.0, population std = 1.0.
2. `official_uniform` (P1 — Official PyTorch nn.Linear Uniform Initialization):
       P_{ij} ~ U(-1 / sqrt(d), +1 / sqrt(d))  where d = input_dim (768 for ViT-B/16)
       Matches `nn.Linear(input_dim, output_dim, bias=False)` (`kaiming_uniform_(a=sqrt(5))`),
       including the exact `(output_dim, input_dim)` sampling orientation transposed to
       `(input_dim, output_dim)`.
       Population variance = 1 / (3 * d), population std = 1 / sqrt(3 * d) (= 1/48 ≈ 0.02083333 for d=768).
3. `gaussian_matched_variance` (P2 — Variance-Matched Gaussian Initialization):
       P_{ij} ~ N(0, 1 / (3 * d))  where d = input_dim
       Population variance = 1 / (3 * d), population std = 1 / sqrt(3 * d) (= 1/48 ≈ 0.02083333 for d=768).
       Matches the exact population variance of `official_uniform` while preserving the Gaussian
       distribution family.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn

VALID_PROJECTION_INITIALIZATIONS = (
    'gaussian_std1',
    'official_uniform',
    'gaussian_matched_variance',
)


class SmoothProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        gaussian_std: float = 1.0,
        initialization: str = 'gaussian_std1',
    ):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or gaussian_std <= 0:
            raise ValueError('Dimensions and gaussian_std must be positive')
        if initialization not in VALID_PROJECTION_INITIALIZATIONS:
            raise ValueError(
                f"Unknown projection initialization '{initialization}'. "
                f"Valid choices: {VALID_PROJECTION_INITIALIZATIONS}"
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.gaussian_std = float(gaussian_std)
        self.initialization = str(initialization)

        # Isolated CPU RNG ensures projection initialization never mutates global RNG state
        generator = torch.Generator(device='cpu').manual_seed(self.seed)

        if self.initialization == 'gaussian_std1':
            self.theoretical_std = float(self.gaussian_std)
            self.theoretical_variance = float(self.gaussian_std ** 2)
            self.uniform_bound: float | None = None
            weight = torch.randn(
                self.input_dim, self.output_dim, generator=generator, dtype=torch.float32
            ) * self.theoretical_std
        elif self.initialization == 'official_uniform':
            bound = 1.0 / math.sqrt(float(self.input_dim))
            self.uniform_bound = float(bound)
            self.theoretical_variance = 1.0 / (3.0 * float(self.input_dim))
            self.theoretical_std = 1.0 / math.sqrt(3.0 * float(self.input_dim))
            # Match nn.Linear(input_dim, output_dim, bias=False) orientation:
            # nn.Linear stores weight as (output_dim, input_dim) and computes x @ weight.T
            raw = torch.empty(self.output_dim, self.input_dim, dtype=torch.float32)
            raw.uniform_(-bound, bound, generator=generator)
            weight = raw.t().contiguous()
        elif self.initialization == 'gaussian_matched_variance':
            self.uniform_bound = None
            self.theoretical_variance = 1.0 / (3.0 * float(self.input_dim))
            self.theoretical_std = 1.0 / math.sqrt(3.0 * float(self.input_dim))
            weight = torch.randn(
                self.input_dim, self.output_dim, generator=generator, dtype=torch.float32
            ) * self.theoretical_std

        # A buffer is saved in checkpoints and *never trained*.
        self.register_buffer('weight', weight)

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute linear projection u = h @ P prior to sigmoid activation."""
        return x @ self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute smooth random projection z = sigmoid(h @ P)."""
        return torch.sigmoid(x @ self.weight)

    @torch.no_grad()
    def get_metadata(self) -> dict[str, Any]:
        """Return auditable metadata and empirical statistics for the frozen projection matrix P."""
        w = self.weight.detach().cpu().float().contiguous()
        sha256_hex = hashlib.sha256(w.numpy().tobytes()).hexdigest()
        return {
            'initialization': self.initialization,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seed': self.seed,
            'theoretical_mean': 0.0,
            'theoretical_std': float(self.theoretical_std),
            'theoretical_variance': float(self.theoretical_variance),
            'uniform_bound': float(self.uniform_bound) if self.uniform_bound is not None else None,
            'measured_mean': float(w.mean().item()),
            'measured_std': float(w.std(unbiased=True).item()),
            'measured_variance': float(w.var(unbiased=True).item()),
            'measured_min': float(w.min().item()),
            'measured_max': float(w.max().item()),
            'matrix_sha256': sha256_hex,
        }
