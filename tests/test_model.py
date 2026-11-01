import torch
from torch import nn
import src.foal.model as module


class DummyFrozenEncoder(nn.Module):
    def __init__(self, name, pretrained=True, feature_fusion=True):
        super().__init__()
        self.feature_fusion = feature_fusion
        self.name = name

    @property
    def feature_dim(self): return 4

    def forward(self, images):
        return images[:, :4]


def test_model_end_to_end_on_synthetic_features(monkeypatch):
    monkeypatch.setattr(module, 'FrozenViT', DummyFrozenEncoder)
    model = module.FOALModel(backbone_name='dummy', projection_dim=5, device='cpu')
    model.classifier.expand_classes(2)
    images = torch.randn(12, 4, generator=torch.Generator().manual_seed(123))
    targets = torch.tensor([0,1]*6)
    model.partial_fit(images[:8], targets[:8])
    model.partial_fit(images[8:], targets[8:])
    assert model(images).shape == (12,2)
    assert model.classifier.samples_seen == 12
    model.classifier.expand_classes(3)
    model.partial_fit(images[:2], torch.tensor([2, 2]))
    assert model(images).shape == (12,3)
    assert all(not p.requires_grad for p in model.projection.parameters())
