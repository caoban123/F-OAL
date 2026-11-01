import torch
from src.foal.projection import SmoothProjection
from src.foal.feature_fusion import fuse_cls_features


def test_projection_is_fixed_and_seeded():
    a = SmoothProjection(4, 6, seed=10)
    b = SmoothProjection(4, 6, seed=10)
    torch.testing.assert_close(a.weight, b.weight)
    assert len(list(a.parameters())) == 0
    x = torch.randn(3, 4)
    assert a(x).shape == (3, 6)
    assert torch.all((a(x) >= 0) & (a(x) <= 1))


def test_feature_fusion_averages_cls_token_across_blocks():
    first = torch.tensor([[[1., 2.], [999., 999.]]])
    second = torch.tensor([[[3., 6.], [888., 888.]]])
    fused = fuse_cls_features([first, second])
    torch.testing.assert_close(fused, torch.tensor([[2., 4.]]))
