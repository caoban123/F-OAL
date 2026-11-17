"""Compose the frozen encoder, optional projection, and analytic classifier."""
from __future__ import annotations
import torch
from torch import nn
from .backbone import FrozenViT
from .projection import SmoothProjection
from .rls import OnlineRidgeRLS


class FOALModel(nn.Module):
    """Forward-only Online Analytic Learning model combining ViT, SmoothProjection, and RLS."""

    def __init__(self, *, backbone_name: str, pretrained: bool = True,
                 feature_fusion: bool = True, smooth_projection: bool = True,
                 projection_dim: int = 1000, projection_seed: int = 42,
                 projection_init: str = 'gaussian_std1', gaussian_std: float = 1.0,
                 gamma: float = 1.0, rls_dtype: str | torch.dtype = 'float64',
                 device: str = 'cpu'):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_fusion_enabled = feature_fusion
        self.smooth_projection_enabled = smooth_projection
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.projection_init = projection_init
        self.gaussian_std = gaussian_std
        self.gamma = gamma
        self.device_str = device
        
        dtype = torch.float64 if rls_dtype in ('float64', torch.float64) else torch.float32
        self.rls_dtype = dtype

        self.encoder = FrozenViT(backbone_name, pretrained=pretrained,
                                 feature_fusion=feature_fusion)
        self.pretrained_tag = getattr(self.encoder, 'pretrained_tag', 'unknown')
        self.pretrained_cfg = getattr(self.encoder, 'pretrained_cfg', {})
        self.projection = SmoothProjection(
            self.encoder.feature_dim,
            projection_dim,
            seed=projection_seed,
            gaussian_std=gaussian_std,
            initialization=projection_init,
        ) if smooth_projection else None
        self.to(device)
        dimension = projection_dim if smooth_projection else self.encoder.feature_dim
        self.classifier = OnlineRidgeRLS(dimension, gamma=gamma, device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through the frozen backbone and optional SmoothProjection.

        Args:
            images: Input image batch tensor of shape (B, C, H, W).

        Returns:
            Encoded feature tensor ready for analytic classification.
        """
        features = self.encoder(images)
        return self.projection(features) if self.projection is not None else features

    @torch.no_grad()
    def partial_fit(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Update the analytic RLS classifier on one labeled mini-batch.

        Args:
            images: Input image batch tensor of shape (B, C, H, W).
            labels: Integer class label tensor of shape (B,).
        """
        self.classifier.update(self.encode(images), labels)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of input images.

        Args:
            images: Input image batch tensor of shape (B, C, H, W).

        Returns:
            Logit score tensor of shape (B, num_classes).
        """
        return self.classifier.scores(self.encode(images))
