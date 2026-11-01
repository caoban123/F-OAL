import json
import tempfile
from pathlib import Path
import pytest
from scripts.summarize_runs import load_run, compute_group_stats, summarize_results


def test_load_and_summarize_runs_protocol_isolation():
    """Verify that pilot, full reproduction, and incomplete runs are strictly separated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        run_full_42 = base_dir / 'foal_seed42_full'
        run_full_123 = base_dir / 'foal_seed123_full'
        run_pilot = base_dir / 'foal_seed42_pilot'
        run_incomplete = base_dir / 'foal_seed999_broken'
        for r in [run_full_42, run_full_123, run_pilot, run_incomplete]:
            r.mkdir()

        # 1. Full Reproduction run (Seed 42, 10 tasks, full data)
        (run_full_42 / 'metadata.json').write_text(json.dumps({
            'method': 'foal',
            'experiment_type': 'REPRODUCTION',
            'training_protocol': {'max_tasks': 10, 'limit_train_per_task': None, 'limit_test_per_task': None},
            'model_config': {'backbone': 'vit_base_patch16_224.augreg_in1k', 'projection_dim': 1000, 'gamma': 1.0},
            'seeds': {'base': 42},
            'device': 'cuda'
        }))
        (run_full_42 / 'metrics.json').write_text(json.dumps({
            'num_tasks': 10,
            'task_matrix': [[0.9] * (i + 1) for i in range(10)],
            'average_incremental_accuracy': 0.90,
            'last_accuracy': 0.88,
            'forgetting': 0.05
        }))
        (run_full_42 / 'task_10.pt').write_bytes(b'DUMMY_CKPT')

        # 2. Full Reproduction run (Seed 123, 10 tasks, full data)
        (run_full_123 / 'metadata.json').write_text(json.dumps({
            'method': 'foal',
            'experiment_type': 'REPRODUCTION',
            'training_protocol': {'max_tasks': 10, 'limit_train_per_task': None, 'limit_test_per_task': None},
            'model_config': {'backbone': 'vit_base_patch16_224.augreg_in1k', 'projection_dim': 1000, 'gamma': 1.0},
            'seeds': {'base': 123},
            'device': 'cuda'
        }))
        (run_full_123 / 'metrics.json').write_text(json.dumps({
            'num_tasks': 10,
            'task_matrix': [[0.92] * (i + 1) for i in range(10)],
            'average_incremental_accuracy': 0.92,
            'last_accuracy': 0.90,
            'forgetting': 0.04
        }))
        (run_full_123 / 'task_10.pt').write_bytes(b'DUMMY_CKPT')

        # 3. Pilot run (Seed 42, 2 tasks, 100 samples)
        (run_pilot / 'metadata.json').write_text(json.dumps({
            'method': 'foal',
            'experiment_type': 'PILOT',
            'training_protocol': {'max_tasks': 2, 'limit_train_per_task': 100, 'limit_test_per_task': 100},
            'model_config': {'backbone': 'vit_base_patch16_224.augreg_in1k', 'projection_dim': 1000, 'gamma': 1.0},
            'seeds': {'base': 42},
            'device': 'cpu'
        }))
        (run_pilot / 'metrics.json').write_text(json.dumps({
            'num_tasks': 2,
            'task_matrix': [[0.78], [0.69, 0.85]],
            'average_incremental_accuracy': 0.775,
            'last_accuracy': 0.77,
            'forgetting': 0.09
        }))
        (run_pilot / 'task_02.pt').write_bytes(b'DUMMY_CKPT')

        # 4. Incomplete run (missing final checkpoint!)
        (run_incomplete / 'metadata.json').write_text(json.dumps({
            'method': 'foal',
            'experiment_type': 'REPRODUCTION',
            'training_protocol': {'max_tasks': 10, 'limit_train_per_task': None, 'limit_test_per_task': None},
            'model_config': {'backbone': 'vit_base_patch16_224.augreg_in1k', 'projection_dim': 1000, 'gamma': 1.0},
            'seeds': {'base': 999},
            'device': 'cuda'
        }))
        (run_incomplete / 'metrics.json').write_text(json.dumps({
            'num_tasks': 10,
            'task_matrix': [[0.5] * (i + 1) for i in range(10)],
            'average_incremental_accuracy': 0.50,
            'last_accuracy': 0.50,
            'forgetting': 0.10
        }))
        # Intentionally NO task_10.pt

        # Run summarizer
        out = summarize_results(results_dir=str(base_dir), output_dir=str(base_dir))
        assert len(out['individual_runs']) == 4

        # Check incomplete status
        incomplete_run = [r for r in out['individual_runs'] if r['run_name'] == 'foal_seed999_broken'][0]
        assert incomplete_run['is_complete'] is False
        assert incomplete_run['status'] == 'INCOMPLETE'
        assert incomplete_run['average_incremental_accuracy'] is None

        # Verify groups are strictly isolated!
        groups = out['groups']
        # Must have separate group for Reproduction (10 tasks) vs Pilot (2 tasks)
        reproduction_group = [g for name, g in groups.items() if 'REPRODUCTION' in name and '10 tasks' in name][0]
        pilot_group = [g for name, g in groups.items() if 'PILOT' in name and '2 tasks' in name][0]

        # Incomplete run must NOT be included in reproduction completed count
        assert reproduction_group['count'] == 2  # Only seed 42 and 123
        assert reproduction_group['total_runs'] == 3  # 2 complete + 1 incomplete
        assert reproduction_group['incomplete_count'] == 1
        assert reproduction_group['average_incremental_accuracy_mean'] == pytest.approx(0.91)
        assert reproduction_group['last_accuracy_mean'] == pytest.approx(0.89)

        # Pilot group must be completely separate with count=1
        assert pilot_group['count'] == 1
        assert pilot_group['average_incremental_accuracy_mean'] == pytest.approx(0.775)


def test_summarize_runs_hyperparameter_and_seed_isolation():
    """Verify that runs differing in feature_fusion, smooth_projection, rls_dtype, or seeds are not grouped together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # Helper to create a run
        def make_run(name, model_extra, seeds_extra):
            r = base_dir / name
            r.mkdir()
            mcfg = {'backbone': 'vit_base_patch16_224.augreg_in1k', 'projection_dim': 1000, 'gamma': 1.0, 'rls_dtype': 'float64'}
            mcfg.update(model_extra)
            (r / 'metadata.json').write_text(json.dumps({
                'method': 'foal',
                'experiment_type': 'REPRODUCTION',
                'training_protocol': {'max_tasks': 10, 'limit_train_per_task': None, 'limit_test_per_task': None},
                'model_config': mcfg,
                'seeds': seeds_extra,
                'device': 'cuda'
            }))
            (r / 'metrics.json').write_text(json.dumps({
                'num_tasks': 10,
                'task_matrix': [[0.9] * (i + 1) for i in range(10)],
                'average_incremental_accuracy': 0.90,
                'last_accuracy': 0.88,
                'forgetting': 0.05
            }))
            (r / 'task_10.pt').write_bytes(b'DUMMY_CKPT')

        # Run 1: Standard coupled seeds, float64, fusion=True, projection=True
        make_run('run_standard', {}, {'base': 42, 'class_order': 42, 'projection': 42, 'stream': 42})
        # Run 2: float32 instead of float64
        make_run('run_float32', {'rls_dtype': 'float32'}, {'base': 42, 'class_order': 42, 'projection': 42, 'stream': 42})
        # Run 3: no fusion
        make_run('run_no_fusion', {'feature_fusion': False}, {'base': 42, 'class_order': 42, 'projection': 42, 'stream': 42})
        # Run 4: no projection
        make_run('run_no_proj', {'smooth_projection': False}, {'base': 42, 'class_order': 42, 'projection': 42, 'stream': 42})
        # Run 5: decoupled seeds (fixed class order 42, projection seed 999)
        make_run('run_decoupled', {}, {'base': 42, 'class_order': 42, 'projection': 999, 'stream': 42})

        out = summarize_results(results_dir=str(base_dir), output_dir=str(base_dir))
        assert len(out['individual_runs']) == 5

        # All 5 runs must be in distinct groups because their protocols differ!
        assert len(out['groups']) == 5


def test_summarize_runs_mixed_none_and_int_limits_and_csv_columns():
    """Verify sorting when limit_test is None vs int, and check summary_aggregated.csv columns."""
    import csv
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        for name, lim_test in [('pilot_all_test', None), ('pilot_100_test', 100)]:
            r = base_dir / name
            r.mkdir()
            (r / 'metadata.json').write_text(json.dumps({
                'method': 'foal',
                'experiment_type': 'PILOT',
                'training_protocol': {'max_tasks': 2, 'limit_train_per_task': 100, 'limit_test_per_task': lim_test},
                'model_config': {'backbone': 'vit_base_patch16_224.augreg_in1k', 'projection_dim': 1000, 'gamma': 1.0},
                'seeds': {'base': 42},
                'device': 'cpu'
            }))
            (r / 'metrics.json').write_text(json.dumps({
                'num_tasks': 2,
                'task_matrix': [[0.8], [0.7, 0.8]],
                'average_incremental_accuracy': 0.775,
                'last_accuracy': 0.75,
                'forgetting': 0.10
            }))
            (r / 'task_02.pt').write_bytes(b'DUMMY_CKPT')

        out = summarize_results(results_dir=str(base_dir), output_dir=str(base_dir))
        assert len(out['groups']) == 2

        with (base_dir / 'summary_aggregated.csv').open('r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]['num_tasks'] == '2'
        assert rows[0]['limit_train'] == '100'
        assert {rows[0]['limit_test'], rows[1]['limit_test']} == {'full', '100'}


