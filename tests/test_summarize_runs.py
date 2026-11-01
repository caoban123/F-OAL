import json
import tempfile
from pathlib import Path
import pytest
from scripts.summarize_runs import load_run, compute_group_stats, summarize_results


def test_load_and_summarize_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        run1 = base_dir / 'foal_seed42'
        run2 = base_dir / 'foal_seed123'
        run1.mkdir(); run2.mkdir()

        # Seed 42 run (tagged SYNTHETIC_TEST_DATA)
        (run1 / 'metadata.json').write_text(json.dumps({
            'method': 'foal',
            'experiment_type': 'SYNTHETIC_TEST_DATA',
            'seeds': {'base': 42},
            'device': 'cpu',
            'git_commit': 'abc123456789'
        }))
        (run1 / 'metrics.json').write_text(json.dumps({
            'num_tasks': 10,
            'average_incremental_accuracy': 0.78,
            'last_accuracy': 0.76,
            'forgetting': 0.08,
            'total_duration_seconds': 120.5
        }))

        # Seed 123 run (tagged SYNTHETIC_TEST_DATA)
        (run2 / 'metadata.json').write_text(json.dumps({
            'method': 'foal',
            'experiment_type': 'SYNTHETIC_TEST_DATA',
            'seeds': {'base': 123},
            'device': 'cpu',
            'git_commit': 'abc123456789'
        }))
        (run2 / 'metrics.json').write_text(json.dumps({
            'num_tasks': 10,
            'average_incremental_accuracy': 0.82,
            'last_accuracy': 0.80,
            'forgetting': 0.06,
            'total_duration_seconds': 122.0
        }))

        # Run summarizer
        out = summarize_results(results_dir=str(base_dir), output_dir=str(base_dir))
        assert len(out['individual_runs']) == 2

        # Check aggregated stats for group
        group_key = 'foal@SYNTHETIC_TEST_DATA'
        assert group_key in out['groups']
        stats = out['groups'][group_key]
        assert stats['count'] == 2
        assert stats['average_incremental_accuracy_mean'] == pytest.approx(0.80)
        assert stats['last_accuracy_mean'] == pytest.approx(0.78)
        assert stats['forgetting_mean'] == pytest.approx(0.07)

        # Check that CSV files were generated
        assert (base_dir / 'summary_runs.csv').exists()
        assert (base_dir / 'summary_aggregated.csv').exists()
