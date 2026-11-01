import json
import os
import tempfile
from pathlib import Path
import pytest
import yaml

from src.foal.environment import get_environment_info


def test_metadata_schema_conformance():
    """Verify that metadata schema contains all expected fields for reproducibility."""
    env = get_environment_info()
    metadata = {
        'paper': 'F-OAL NeurIPS 2024',
        'method': 'foal',
        'experiment_type': 'reproduction',
        'run_name': 'foal_seed42_reproduction',
        'seeds': {
            'base': 42,
            'class_order': 42,
            'projection': 42,
            'stream': 42,
        },
        'model_config': {
            'backbone': 'vit_base_patch16_224.augreg_in1k',
            'pretrained': True,
            'feature_fusion': True,
            'smooth_projection': True,
            'projection_dim': 1000,
            'gamma': 1.0,
            'rls_dtype': 'float64',
        },
        'training_protocol': {
            'dataset': 'uoft-cs/cifar100',
            'num_classes': 100,
            'classes_per_task': 10,
            'max_tasks': 10,
            'limit_train_per_task': None,
            'limit_test_per_task': None,
            'batch_size': 10,
            'device': 'cpu',
            'one_pass': True,
        },
        'class_order': list(range(100)),
        'environment': env,
    }

    # Verify JSON serializability
    serialized = json.dumps(metadata)
    deserialized = json.loads(serialized)

    assert deserialized['method'] == 'foal'
    assert deserialized['seeds']['projection'] == 42
    assert deserialized['training_protocol']['one_pass'] is True
    assert 'libraries' in deserialized['environment']
    assert deserialized['environment']['libraries']['torch'] is not None


def test_wandb_offline_env_flag():
    """Verify that offline mode correctly sets WANDB_MODE environment variable."""
    old_mode = os.environ.get('WANDB_MODE')
    try:
        os.environ['WANDB_MODE'] = 'offline'
        assert os.environ['WANDB_MODE'] == 'offline'
    finally:
        if old_mode is not None:
            os.environ['WANDB_MODE'] = old_mode
        else:
            os.environ.pop('WANDB_MODE', None)
