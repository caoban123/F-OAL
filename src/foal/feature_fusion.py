"""Fuse intermediate ViT block representations with an arithmetic mean."""
from __future__ import annotations
import torch


def fuse_cls_features(block_features: list[torch.Tensor]) -> torch.Tensor:
    """Input: each [batch, tokens, embedding_dim], CLS at token index 0.

    Equivalent to first averaging complete block tensors and then taking CLS.
    """
    if not block_features:
        raise ValueError('At least one ViT block representation is required')
    if any(f.ndim != 3 or f.shape != block_features[0].shape for f in block_features):
        raise ValueError('Inconsistent ViT feature shapes')
    return torch.stack([f[:, 0, :] for f in block_features]).mean(dim=0)
