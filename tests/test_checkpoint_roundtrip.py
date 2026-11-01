import tempfile
from pathlib import Path
import torch
import pytest
from src.foal.model import FOALModel


def test_checkpoint_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / 'task_01.pt'
        
        # Initialize model
        model = FOALModel(backbone_name='vit_base_patch16_224.augreg_in1k',
                          pretrained=True, feature_fusion=True,
                          smooth_projection=True, projection_dim=128,
                          projection_seed=42, gamma=1.0, rls_dtype='float64')
        model.classifier.expand_classes(3)
        dummy_x = torch.randn(6, 3, 224, 224)
        dummy_y = torch.tensor([0, 1, 2, 0, 1, 2])
        model.partial_fit(dummy_x, dummy_y)
        
        # Raw continuous class scores (logits) before saving
        scores_before = model(dummy_x)
        preds_before = scores_before.argmax(dim=1)
        
        # Save checkpoint exactly as done in scripts/train.py
        torch.save({
            'model_config': {
                'backbone_name': 'vit_base_patch16_224.augreg_in1k',
                'feature_fusion': True,
                'smooth_projection': True,
                'projection_dim': 128,
                'projection_seed': 42,
                'gamma': 1.0,
                'rls_dtype': 'float64'
            },
            'projection_state': model.projection.state_dict(),
            'classifier_state': model.classifier.state_dict(),
            'class_order': list(range(100)),
            'task_number': 1,
            'config': {'dataset': {'classes_per_task': 10, 'image_size': 224}, 'training': {'batch_size': 10}}
        }, ckpt_path)
        
        # Reload checkpoint exactly as done in scripts/evaluate.py
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        m = ckpt['model_config'].copy()
        model_reloaded = FOALModel(**m, pretrained=True)
        model_reloaded.projection.load_state_dict(ckpt['projection_state'])
        model_reloaded.classifier.load_state_dict(ckpt['classifier_state'])
        model_reloaded.eval()
        
        # Raw continuous class scores after reloading
        scores_after = model_reloaded(dummy_x)
        preds_after = scores_after.argmax(dim=1)
        
        # Exact numerical comparison of scores (logits) and predictions
        torch.testing.assert_close(scores_before, scores_after, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(preds_before, preds_after)
