import json
import tempfile
from pathlib import Path
import pytest
import torch

from src.foal.datasets import make_class_order
from src.foal.projection import SmoothProjection
from scripts.run_seeds import is_run_completed
from src.foal.metrics import aggregate_multi_seed_metrics


def test_seed_separation_class_order():
    """Verify that class order depends strictly on class_order_seed."""
    order_a1 = make_class_order(100, seed=42)
    order_a2 = make_class_order(100, seed=42)
    order_b = make_class_order(100, seed=123)

    assert order_a1 == order_a2
    assert order_a1 != order_b
    assert len(order_a1) == 100
    assert len(set(order_a1)) == 100


def test_seed_separation_projection():
    """Verify that SmoothProjection initialization depends strictly on projection_seed."""
    proj_a1 = SmoothProjection(input_dim=768, output_dim=1000, seed=42)
    proj_a2 = SmoothProjection(input_dim=768, output_dim=1000, seed=42)
    proj_b = SmoothProjection(input_dim=768, output_dim=1000, seed=123)

    # Identical seed produces identical weight matrix
    torch.testing.assert_close(proj_a1.weight, proj_a2.weight)

    # Different seed produces different weight matrix
    diff = (proj_a1.weight - proj_b.weight).abs().max()
    assert diff > 0.1


def test_is_run_completed_logic():
    """Verify skip-completed detection logic requires metrics, final checkpoint, and matching protocol."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / 'test_run'
        assert not is_run_completed(run_dir)

        run_dir.mkdir()
        assert not is_run_completed(run_dir)

        # Write invalid/partial files
        (run_dir / 'metadata.json').write_text(json.dumps({
            'seeds': {'base': 42},
            'training_protocol': {'max_tasks': 2, 'limit_train_per_task': 100, 'limit_test_per_task': 100},
            'model_config': {'feature_fusion': True, 'smooth_projection': True}
        }))
        assert not is_run_completed(run_dir)

        # Write metrics but WITHOUT final checkpoint
        (run_dir / 'metrics.json').write_text(json.dumps({
            'task_matrix': [[0.78], [0.69, 0.85]],
            'average_incremental_accuracy': 0.775,
            'last_accuracy': 0.77,
            'forgetting': 0.09
        }))
        # Fails because task_02.pt does not exist
        assert not is_run_completed(run_dir, expected_protocol={'max_tasks': 2, 'seed': 42})

        # Add final checkpoint
        (run_dir / 'task_02.pt').write_bytes(b'CHECKPOINT_DATA')

        # Matches exact protocol
        valid_proto = {
            'max_tasks': 2,
            'seed': 42,
            'limit_train_per_task': 100,
            'limit_test_per_task': 100,
            'feature_fusion': True,
            'smooth_projection': True
        }
        assert is_run_completed(run_dir, expected_protocol=valid_proto)

        # Should FAIL if requested seed is different (e.g. 123 instead of 42)
        assert not is_run_completed(run_dir, expected_protocol={**valid_proto, 'seed': 123})

        # Should FAIL if requested tasks is different (e.g. 5 instead of 2)
        assert not is_run_completed(run_dir, expected_protocol={**valid_proto, 'max_tasks': 5})

        # Should FAIL if requested limit is different (e.g. full data instead of 100)
        assert not is_run_completed(run_dir, expected_protocol={**valid_proto, 'limit_train_per_task': None})
