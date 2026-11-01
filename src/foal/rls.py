"""Block Recursive Least Squares for F-OAL (paper Equations 10–13).

The method retains sufficient statistics via R and W without replaying samples.
This is a clean-room implementation based on the published equations.
"""
from __future__ import annotations

import torch


class OnlineRidgeRLS:
    """Sequential ridge regression with expanding class dimension.

    R = (gamma * I + H_seen.T @ H_seen)^(-1)
    W = R @ H_seen.T @ one_hot(y_seen)

    The class ordering must be contiguous: 0, 1, ... C-1. A class is
    introduced before receiving its first batch; older class columns remain.
    """

    def __init__(self, feature_dim: int, gamma: float = 1.0,
                 device: str | torch.device = 'cpu', dtype: torch.dtype = torch.float64):
        if feature_dim < 1 or gamma <= 0:
            raise ValueError('feature_dim >= 1 and gamma > 0 are required')
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.dtype = dtype
        self.R = torch.eye(feature_dim, device=self.device, dtype=dtype) / gamma
        self.W = torch.zeros((feature_dim, 0), device=self.device, dtype=dtype)
        self.samples_seen = 0

    @property
    def num_classes(self) -> int:
        return self.W.shape[1]

    @torch.no_grad()
    def expand_classes(self, total_classes: int) -> None:
        """Add output columns without changing the previous R matrix."""
        if total_classes < self.num_classes:
            raise ValueError('Cannot remove previously seen classes')
        new = total_classes - self.num_classes
        if new:
            self.W = torch.cat((self.W, torch.zeros(
                (self.feature_dim, new), device=self.device, dtype=self.dtype
            )), dim=1)

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Woodbury update on a single mini-batch, using no previous samples."""
        H = features.to(device=self.device, dtype=self.dtype)
        y = labels.to(device=self.device, dtype=torch.long)
        if H.ndim != 2 or H.shape[1] != self.feature_dim:
            raise ValueError(f'Expected features [batch, {self.feature_dim}]')
        if y.ndim != 1 or y.shape[0] != H.shape[0] or H.shape[0] == 0:
            raise ValueError('Expected non-empty labels [batch]')
        if self.num_classes == 0 or y.min() < 0 or y.max() >= self.num_classes:
            raise ValueError('Call expand_classes before updating with new labels')

        # K = R_old H.T; A = I + H R_old H.T  (batch-sized solve).
        K = self.R @ H.T
        A = torch.eye(H.shape[0], device=self.device, dtype=self.dtype) + H @ K
        # torch.linalg.solve avoids explicitly constructing A^(-1).
        gain = torch.linalg.solve(A, K.T).T
        Y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(self.dtype)
        # Using the equivalent form gain = R_new H.T.
        residual = Y - H @ self.W
        self.W = self.W + gain @ residual
        self.R = self.R - gain @ K.T
        self.R = 0.5 * (self.R + self.R.T)  # reduce FP roundoff asymmetry
        self.samples_seen += H.shape[0]

    @torch.no_grad()
    def scores(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 0:
            raise RuntimeError('No classes have been seen yet')
        return features.to(device=self.device, dtype=self.dtype) @ self.W

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.scores(features).argmax(dim=1)

    def state_dict(self) -> dict:
        return dict(feature_dim=self.feature_dim, gamma=self.gamma,
                    R=self.R, W=self.W, samples_seen=self.samples_seen)

    def load_state_dict(self, state: dict) -> None:
        if int(state['feature_dim']) != self.feature_dim:
            raise ValueError('Mismatched feature dimension')
        self.R = state['R'].to(self.device, self.dtype)
        self.W = state['W'].to(self.device, self.dtype)
        self.samples_seen = int(state['samples_seen'])
