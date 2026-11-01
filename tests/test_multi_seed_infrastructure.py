import json
import tempfile
from pathlib import Path
import pytest
import torch

from src.foal.datasets import make_class_order
from src.foal.projection import SmoothProjection
from scripts.run_seeds import is_run_completed
from scripts.train import resolve_seed
from src.foal.metrics import aggregate_multi_seed_metrics
import yaml


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

        # Should FAIL if requested model config differs (e.g. feature_fusion=False)
        assert not is_run_completed(run_dir, expected_protocol={**valid_proto, 'feature_fusion': False})

        # Should FAIL if projection_seed or class_order_seed is different
        assert not is_run_completed(run_dir, expected_protocol={**valid_proto, 'projection_seed': 999})


def test_resolve_seed_fallback_and_null_safety():
    """Verify seed resolution handles CLI args, YAML null, and defaults safely."""
    # CLI takes highest precedence
    assert resolve_seed(cli_value=123, config_value=456, default_seed=789) == 123
    assert resolve_seed(cli_value='123', config_value=None, default_seed=789) == 123

    # Config takes precedence over default when CLI is None
    assert resolve_seed(cli_value=None, config_value=456, default_seed=789) == 456
    assert resolve_seed(cli_value=None, config_value='456', default_seed=789) == 456

    # YAML null (None in Python) falls back to default_seed safely without TypeError
    assert resolve_seed(cli_value=None, config_value=None, default_seed=789) == 789
    assert resolve_seed(cli_value=None, config_value=None, default_seed=42) == 42

    # Verify directly with configs/cifar100.yaml content
    cfg = yaml.safe_load(Path('configs/cifar100.yaml').read_text(encoding='utf-8'))
    assert cfg.get('class_order_seed') is None
    assert cfg.get('projection_seed') is None
    assert cfg.get('stream_seed') is None

    base_seed = resolve_seed(None, cfg.get('seed'), 42)
    assert base_seed == 42
    assert resolve_seed(None, cfg.get('class_order_seed'), base_seed) == 42
    assert resolve_seed(None, cfg.get('projection_seed'), base_seed) == 42
    assert resolve_seed(None, cfg.get('stream_seed'), base_seed) == 42

    # When user overrides with CLI --seed 123
    base_seed_custom = resolve_seed(123, cfg.get('seed'), 42)
    assert base_seed_custom == 123
    assert resolve_seed(None, cfg.get('class_order_seed'), base_seed_custom) == 123
    assert resolve_seed(None, cfg.get('projection_seed'), base_seed_custom) == 123
    assert resolve_seed(None, cfg.get('stream_seed'), base_seed_custom) == 123


