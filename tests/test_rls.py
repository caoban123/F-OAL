import pytest
import torch
from src.foal.rls import OnlineRidgeRLS


def test_rls_matches_closed_form_ridge_with_expanding_classes():
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(27, 7, generator=gen, dtype=torch.float64)
    # Classes 0/1 first, then class 2 later; no sample is reused.
    y = torch.tensor([0, 1] * 7 + [2, 0, 1] * 4 + [2], dtype=torch.long)
    gamma = 0.7
    model = OnlineRidgeRLS(7, gamma=gamma)
    model.expand_classes(2)
    model.update(x[:5], y[:5]); model.update(x[5:14], y[5:14])
    model.expand_classes(3)
    model.update(x[14:21], y[14:21]); model.update(x[21:], y[21:])
    target = torch.nn.functional.one_hot(y, num_classes=3).double()
    reference = torch.linalg.solve(x.T @ x + gamma * torch.eye(7, dtype=x.dtype), x.T @ target)
    torch.testing.assert_close(model.W, reference, atol=1e-11, rtol=1e-11)
    expected_R = torch.linalg.inv(x.T @ x + gamma * torch.eye(7, dtype=x.dtype))
    torch.testing.assert_close(model.R, expected_R, atol=1e-11, rtol=1e-11)
    assert model.samples_seen == len(x)


def test_can_only_add_classes_not_remove():
    model = OnlineRidgeRLS(3)
    model.expand_classes(2)
    with pytest.raises(ValueError, match='remove'):
        model.expand_classes(1)
    with pytest.raises(ValueError, match='expand_classes'):
        model.update(torch.randn(2, 3), torch.tensor([1, 2]))


def test_batch_partition_does_not_change_exact_solution():
    x = torch.randn(30, 4, generator=torch.Generator().manual_seed(77), dtype=torch.float64)
    y = torch.tensor([0, 1, 2] * 10)
    a = OnlineRidgeRLS(4); b = OnlineRidgeRLS(4)
    a.expand_classes(3); b.expand_classes(3)
    a.update(x, y)
    for i in range(0, len(y), 5): b.update(x[i:i+5], y[i:i+5])
    torch.testing.assert_close(a.W, b.W, atol=1e-11, rtol=1e-11)


def test_nan_or_inf_detection():
    model = OnlineRidgeRLS(4)
    model.expand_classes(2)
    bad_x = torch.tensor([[1.0, float('nan'), 2.0, 3.0]])
    bad_y = torch.tensor([0])
    with pytest.raises(ValueError, match='NaN or Inf'):
        model.update(bad_x, bad_y)
