import torch
import pytest
from src.foal.model import FOALModel
from src.foal.rls import OnlineRidgeRLS
from src.foal.datasets import load_cifar100, cifar100_transforms


def test_rls_matches_ridge_on_real_vit_features():
    train_ds, _ = load_cifar100()
    trans = cifar100_transforms(size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    # Take 16 real samples from CIFAR-100
    images = torch.stack([trans(train_ds[i]['img'].convert('RGB')) for i in range(16)])
    raw_labels = torch.tensor([int(train_ds[i]['fine_label']) for i in range(16)])
    
    # Map labels to 0..C-1
    unique_labels = sorted(set(raw_labels.tolist()))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_map[l.item()] for l in raw_labels], dtype=torch.long)
    num_classes = len(unique_labels)
    
    # Encode with frozen ViT + SmoothProjection
    model = FOALModel(backbone_name='vit_base_patch16_224.augreg_in1k',
                      pretrained=True, feature_fusion=True,
                      smooth_projection=True, projection_dim=512,
                      projection_seed=42, gamma=1.0, rls_dtype='float64')
    
    with torch.no_grad():
        H = model.encode(images)  # shape (16, 512)
    
    assert H.shape == (16, 512)
    assert torch.all(torch.isfinite(H))
    
    # Sequential online RLS update in mini-batches of size 4
    rls = OnlineRidgeRLS(feature_dim=512, gamma=1.0, dtype=torch.float64)
    rls.expand_classes(num_classes)
    
    batch_size = 4
    for i in range(0, len(H), batch_size):
        rls.update(H[i:i+batch_size], mapped_labels[i:i+batch_size])
        
    # Closed-form batch Ridge Regression solution
    H_d = H.double()
    Y_onehot = torch.nn.functional.one_hot(mapped_labels, num_classes=num_classes).double()
    A = H_d.T @ H_d + 1.0 * torch.eye(512, dtype=torch.float64)
    W_batch = torch.linalg.solve(A, H_d.T @ Y_onehot)
    R_batch = torch.linalg.inv(A)
    
    # Verify mathematical equivalence
    torch.testing.assert_close(rls.W, W_batch, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(rls.R, R_batch, atol=1e-10, rtol=1e-10)
