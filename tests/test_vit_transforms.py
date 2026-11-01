import torch
import pytest
from src.foal.datasets import cifar100_transforms
from src.foal.backbone import FrozenViT


def test_vit_transforms_structure():
    t = cifar100_transforms(size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert len(t.transforms) == 3
    # Check normalize values
    norm = t.transforms[-1]
    assert tuple(norm.mean) == (0.5, 0.5, 0.5)
    assert tuple(norm.std) == (0.5, 0.5, 0.5)


def test_frozen_vit_properties():
    encoder = FrozenViT(name='vit_base_patch16_224.augreg_in1k', pretrained=True, feature_fusion=True)
    assert encoder.feature_dim == 768
    # Assert parameters do not require grad
    for p in encoder.parameters():
        assert not p.requires_grad
    # Assert mode is eval even when train() is called
    encoder.train(True)
    assert not encoder.network.training

    # Test forward with dummy image batch [2, 3, 224, 224]
    dummy = torch.randn(2, 3, 224, 224)
    feat = encoder(dummy)
    assert feat.shape == (2, 768)
    assert torch.all(torch.isfinite(feat))


def test_feature_fusion_applies_final_layernorm():
    # Regression test: verify that final LayerNorm (network.norm) is applied
    enc_no_ff = FrozenViT(name='vit_base_patch16_224.augreg_in1k', pretrained=False, feature_fusion=False)
    x = torch.randn(2, 3, 224, 224)
    out_no_ff = enc_no_ff(x)
    out_timm = enc_no_ff.network.forward_features(x)[:, 0, :]
    torch.testing.assert_close(out_no_ff, out_timm, atol=1e-6, rtol=1e-6)

    enc_ff = FrozenViT(name='vit_base_patch16_224.augreg_in1k', pretrained=False, feature_fusion=True)
    out_ff = enc_ff(x)
    assert out_ff.shape == (2, 768)
    assert torch.all(torch.isfinite(out_ff))
