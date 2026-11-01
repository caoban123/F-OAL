"""Frozen timm Vision Transformer, with block-output feature fusion.

The actual model and weight checkpoint are configurable; matching the author's
exact weights/preprocessing must be validated before claiming a reproduction.
"""
from __future__ import annotations
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

    def train(self, mode: bool = True):
        # A frozen model must remain in eval mode even if its parent is trained.
        super().train(False)
        self.network.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
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
        # For ablation, take the last block at the same pre-normalization level
        # as the full fusion; do not introduce a final layernorm confound.
        return fuse_cls_features(collected if self.feature_fusion else collected[-1:])
