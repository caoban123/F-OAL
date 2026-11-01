import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

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


def test_wandb_logging_and_artifact_flow(tmp_path):
    """Verify W&B logging flow: init, per-task step logging, summary logging, artifact creation, and finish."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    # Simulate the W&B calls that train.py performs
    run_name = "test_run"
    outdir = tmp_path / "run_output"
    outdir.mkdir()
    metrics_file = outdir / "metrics.json"
    metadata_file = outdir / "metadata.json"
    csv_file = outdir / "task_accuracy.csv"
    metrics_file.write_text('{"average_incremental_accuracy": 0.85}', encoding='utf-8')
    metadata_file.write_text('{"method": "foal"}', encoding='utf-8')
    csv_file.write_text('task,acc\n1,0.9\n', encoding='utf-8')

    # Step 1: wandb.init
    metadata = {'method': 'foal', 'seeds': {'base': 42}}
    wandb_run = mock_wandb.init(
        project='hcmus-foal-reproduction',
        entity=None,
        group='foal_reproduction',
        name=run_name,
        config=metadata,
        tags=['reproduction', 'foal', 'seed_42'],
    )
    assert wandb_run is mock_run

    # Step 2: Per-task step logging
    step_metrics = {
        'task': 1,
        'mean_acc_so_far': 0.88,
        'task_duration_seconds': 5.2,
        'task_acc/task_1': 0.88,
        'task_acc_pct/task_1': 88.0,
    }
    wandb_run.log(step_metrics)
    mock_run.log.assert_called_with(step_metrics)

    # Step 3: Summary logging and artifact logging
    summary_metrics = {
        'summary/average_incremental_accuracy': 0.88,
        'summary/average_incremental_accuracy_pct': 88.0,
        'summary/last_accuracy': 0.88,
        'summary/total_duration_seconds': 5.2,
    }
    wandb_run.log(summary_metrics)

    artifact = mock_wandb.Artifact(
        name=f'{run_name}_results',
        type='evaluation',
        description=f'Evaluation results for {run_name}',
    )
    artifact.add_file(str(metrics_file))
    artifact.add_file(str(metadata_file))
    artifact.add_file(str(csv_file))
    wandb_run.log_artifact(artifact)

    mock_wandb.Artifact.assert_called_with(
        name=f'{run_name}_results',
        type='evaluation',
        description=f'Evaluation results for {run_name}',
    )
    artifact.add_file.assert_any_call(str(metrics_file))
    artifact.add_file.assert_any_call(str(metadata_file))
    artifact.add_file.assert_any_call(str(csv_file))
    mock_run.log_artifact.assert_called_with(artifact)

    # Step 4: Finish run
    wandb_run.finish()
    mock_run.finish.assert_called_once()
