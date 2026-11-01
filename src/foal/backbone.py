"""Frozen timm Vision Transformer, with block-output feature fusion.

The backbone parameters are strictly frozen (requires_grad=False) and kept
in eval mode.
Feature fusion averages the CLS tokens across all transformer blocks,
and then applies the final LayerNorm (network.norm), matching the official
ViT representation specification.
"""
from __future__ import annotations
from typing import Any, Dict
import torch
from torch import nn
from .feature_fusion import fuse_cls_features


class FrozenViT(nn.Module):
    def __init__(self, name: str = 'vit_base_patch16_224.augreg_in1k',
                 pretrained: bool = True, feature_fusion: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError('Install optional training requirements: pip install timm') from exc
        self.model_name = name
        self.network = timm.create_model(name, pretrained=pretrained, num_classes=0)
        if not hasattr(self.network, 'blocks'):
            raise TypeError('This encoder requires a timm ViT with .blocks')
        self.feature_fusion = feature_fusion
        for p in self.network.parameters():
            p.requires_grad_(False)
        self.network.eval()

    @property
    def feature_dim(self) -> int:
        return int(self.network.num_features)

    @property
    def default_cfg(self) -> Dict[str, Any]:
        """Return the default data configuration from timm (mean, std, size, interpolation)."""
        import timm.data
        return timm.data.resolve_model_data_config(self.network)

    def train(self, mode: bool = True):
        # A frozen model must remain in eval mode even if its parent is trained.
        super().train(False)
        self.network.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f'Expected images tensor of shape [batch, channels, height, width], got {images.shape}')
        self.network.eval()
        collected: list[torch.Tensor] = []
        hooks = []
        def save_block(_module, _inputs, output):
            collected.append(output if isinstance(output, torch.Tensor) else output[0])
        try:
            for block in self.network.blocks:
                hooks.append(block.register_forward_hook(save_block))
            _ = self.network.forward_features(images)
        finally:
            for hook in hooks:
                hook.remove()
        
        # When feature_fusion is True, average CLS tokens across all blocks.
        # When False (ablation), take the last block's CLS token.
        fused = fuse_cls_features(collected if self.feature_fusion else collected[-1:])
        
        # Apply the backbone's final LayerNorm to the fused representation
        if hasattr(self.network, 'norm') and self.network.norm is not None:
            fused = self.network.norm(fused)
            
        return fused
