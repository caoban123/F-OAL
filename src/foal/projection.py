"""Fixed Gaussian projection followed by sigmoid (F-OAL Eq. 1)."""
from __future__ import annotations

import torch
from torch import nn


class SmoothProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seed: int = 42,
                 gaussian_std: float = 1.0):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or gaussian_std <= 0:
            raise ValueError('Dimensions and gaussian_std must be positive')
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        weight = torch.randn(input_dim, output_dim, generator=generator) * gaussian_std
        # A buffer is saved in checkpoints and *never trained*.
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.weight)
