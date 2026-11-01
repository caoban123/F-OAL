"""Multi-seed experiment orchestration for reproducible continual learning evaluations.

Usage:
    python -m scripts.run_seeds --seeds 42 123 2026 --config configs/cifar100.yaml --run-name-prefix foal_3seeds
    python -m scripts.run_seeds --seeds 42 123 --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100 --skip-completed
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import yaml
from src.foal.metrics import aggregate_multi_seed_metrics, format_percentage
from scripts.train import run as train_run


def is_run_completed(run_dir: Path, expected_protocol: dict[str, Any] | None = None) -> bool:
    """Check if a run directory contains valid metrics, final checkpoint, and matching protocol."""
    metrics_file = run_dir / 'metrics.json'
    metadata_file = run_dir / 'metadata.json'
    if not (metrics_file.exists() and metadata_file.exists()):
        return False
    try:
        metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        
        # 1. Required metric keys
        if ('average_incremental_accuracy' not in metrics or
                'last_accuracy' not in metrics or
                'forgetting' not in metrics):
            return False
            
        task_matrix = metrics.get('task_matrix', [])
        if not task_matrix:
            return False

        # 2. Number of tasks
        expected_tasks = expected_protocol.get('max_tasks') if expected_protocol else None
        if expected_tasks is not None:
            if len(task_matrix) != expected_tasks:
                return False
            recorded_tasks = metadata.get('training_protocol', {}).get('max_tasks')
            if recorded_tasks != expected_tasks:
                return False
        else:
            expected_tasks = len(task_matrix)

        # 3. Final checkpoint must exist and be non-empty
        final_ckpt = run_dir / f'task_{expected_tasks:02d}.pt'
        if not (final_ckpt.exists() and final_ckpt.stat().st_size > 0):
            return False

        # 4. If expected_protocol is given, verify strict parameter matching
        if expected_protocol is not None:
            # Seed match
            req_seed = expected_protocol.get('seed')
            if req_seed is not None:
                recorded_seed = metadata.get('seeds', {}).get('base', metadata.get('config', {}).get('seed'))
                if recorded_seed != req_seed:
                    return False
            
            # Protocol match
            proto = metadata.get('training_protocol', {})
            if 'limit_train_per_task' in expected_protocol:
                if proto.get('limit_train_per_task') != expected_protocol['limit_train_per_task']:
                    return False
            if 'limit_test_per_task' in expected_protocol:
                if proto.get('limit_test_per_task') != expected_protocol['limit_test_per_task']:
                    return False

            # Model config match
            model_cfg = metadata.get('model_config', {})
            if 'feature_fusion' in expected_protocol:
                if model_cfg.get('feature_fusion') != expected_protocol['feature_fusion']:
                    return False
            if 'smooth_projection' in expected_protocol:
                if model_cfg.get('smooth_projection') != expected_protocol['smooth_projection']:
                    return False

        return True
    except Exception:
        return False


def run_multi_seed(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    results_base = Path(args.results_dir or cfg.get('logging', {}).get('results_dir', 'results'))
    results_base.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    if not seeds:
        seeds = [42]

    prefix = args.run_name_prefix or 'foal_multiseed'
    runs_results: list[dict[str, Any]] = []

    print(f'Starting multi-seed execution over {len(seeds)} seed(s): {seeds}')
    print(f'Results will be saved under: {results_base}')

    for idx, seed in enumerate(seeds, start=1):
        run_name = f'{prefix}_seed{seed}'
        run_dir = results_base / run_name
        print(f"\n[{idx}/{len(seeds)}] Preparing run for seed={seed}: '{run_name}'")

        num_classes = int(cfg.get('dataset', {}).get('num_classes', 100))
        classes_per_task = int(cfg.get('dataset', {}).get('classes_per_task', 10))
        expected_tasks = args.max_tasks or (num_classes // classes_per_task)

        expected_protocol = {
            'seed': seed,
            'max_tasks': expected_tasks,
            'limit_train_per_task': args.limit_train_per_task,
            'limit_test_per_task': args.limit_test_per_task,
            'feature_fusion': bool(cfg.get('model', {}).get('feature_fusion', True)) and not args.no_feature_fusion,
            'smooth_projection': bool(cfg.get('model', {}).get('smooth_projection', True)) and not args.no_projection,
        }

        if args.skip_completed and is_run_completed(run_dir, expected_protocol=expected_protocol):
            print(f"Skipping already completed run: '{run_dir}'")
            saved_metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
            runs_results.append({
                'seed': seed,
                'run_name': run_name,
                'metrics': saved_metrics,
                'status': 'SKIPPED_EXISTING',
            })
            continue

        train_args = argparse.Namespace(
            config=args.config,
            max_tasks=args.max_tasks,
            limit_train_per_task=args.limit_train_per_task,
            limit_test_per_task=args.limit_test_per_task,
            seed=seed,
            class_order_seed=args.class_order_seed if args.class_order_seed is not None else seed,
            projection_seed=args.projection_seed if args.projection_seed is not None else seed,
            stream_seed=args.stream_seed if args.stream_seed is not None else seed,
            device=args.device,
            experiment_type=args.experiment_type,
            no_feature_fusion=args.no_feature_fusion,
            no_projection=args.no_projection,
            run_name=run_name,
            wandb=args.wandb,
            wandb_offline=args.wandb_offline,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=prefix,
            wandb_tags=f'multiseed,{prefix}' + (f',{args.wandb_tags}' if args.wandb_tags else ''),
        )

        run_metrics = train_run(train_args)
        runs_results.append({
            'seed': seed,
            'run_name': run_name,
            'metrics': run_metrics,
            'status': 'COMPLETED',
        })

    # Aggregate metrics across seeds
    all_metrics = [r['metrics'] for r in runs_results]
    aggregated = aggregate_multi_seed_metrics(all_metrics)

    summary_output = {
        'run_name_prefix': prefix,
        'seeds': seeds,
        'aggregated': aggregated,
        'runs': [
            {
                'seed': r['seed'],
                'run_name': r['run_name'],
                'status': r['status'],
                'average_incremental_accuracy': r['metrics'].get('average_incremental_accuracy'),
                'last_accuracy': r['metrics'].get('last_accuracy'),
                'forgetting': r['metrics'].get('forgetting'),
            }
            for r in runs_results
        ],
    }

    # Save summary files
    summary_dir = results_base / prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = summary_dir / 'summary_multiseed.json'
    summary_csv_path = summary_dir / 'summary_multiseed.csv'

    summary_json_path.write_text(json.dumps(summary_output, indent=2), encoding='utf-8')

    with summary_csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'run_name', 'status', 'avg_inc_acc', 'last_acc', 'forgetting'])
        for r in runs_results:
            m = r['metrics']
            writer.writerow([
                r['seed'],
                r['run_name'],
                r['status'],
                m.get('average_incremental_accuracy'),
                m.get('last_accuracy'),
                m.get('forgetting'),
            ])
        writer.writerow([])
        writer.writerow(['STATISTIC', 'SEEDS_COUNT', '', 'AVG_INC_ACC', 'LAST_ACC', 'FORGETTING'])
        writer.writerow([
            'MEAN +/- STD',
            len(seeds),
            '',
            aggregated.get('average_incremental_accuracy_display'),
            aggregated.get('last_accuracy_display'),
            aggregated.get('forgetting_display'),
        ])

    # Print summary report
    print('\n' + '=' * 70)
    print(f'MULTI-SEED EXPERIMENT SUMMARY — {prefix}')
    print('=' * 70)
    print(f'Seeds evaluated : {seeds} (N={len(seeds)})')
    print(f"Avg Incr Acc    : {aggregated.get('average_incremental_accuracy_display')}")
    print(f"Last Accuracy   : {aggregated.get('last_accuracy_display')}")
    print(f"Forgetting Rate : {aggregated.get('forgetting_display')}")
    print(f'Saved JSON to   : {summary_json_path}')
    print(f'Saved CSV to    : {summary_csv_path}')
    print('=' * 70 + '\n')

    return summary_output


def main():
    p = argparse.ArgumentParser(description='Multi-seed F-OAL experiment runner.')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026],
                   help='List of random seeds to evaluate sequentially')
    p.add_argument('--config', default='configs/cifar100.yaml',
                   help='Path to configuration YAML')
    p.add_argument('--run-name-prefix', default='foal_multiseed',
                   help='Prefix for run subdirectories and W&B group')
    p.add_argument('--max-tasks', type=int, default=None,
                   help='Pilot only: limit number of sequential tasks')
    p.add_argument('--limit-train-per-task', type=int, default=None,
                   help='Pilot only: limit training samples per task')
    p.add_argument('--limit-test-per-task', type=int, default=None,
                   help='Pilot only: limit test samples evaluated per task')
    p.add_argument('--class-order-seed', type=int, default=None,
                   help='Fix class order across all seed runs if specified')
    p.add_argument('--projection-seed', type=int, default=None,
                   help='Fix projection seed across all seed runs if specified')
    p.add_argument('--stream-seed', type=int, default=None,
                   help='Fix stream shuffling seed across all seed runs if specified')
    p.add_argument('--device', default=None,
                   help="Device ('cpu', 'cuda')")
    p.add_argument('--experiment-type', default=None,
                   choices=['pilot', 'reproduction', 'ablation', 'additional-analysis'])
    p.add_argument('--no-feature-fusion', action='store_true',
                   help='Ablation: disable feature fusion')
    p.add_argument('--no-projection', action='store_true',
                   help='Ablation: disable smooth projection')
    p.add_argument('--skip-completed', action='store_true',
                   help='Skip runs that already have complete metrics on disk')
    p.add_argument('--results-dir', default='results',
                   help='Directory to store experiment results')
    p.add_argument('--wandb', action='store_true',
                   help='Enable Weights & Biases logging')
    p.add_argument('--wandb-offline', action='store_true',
                   help='Enable Weights & Biases in offline mode')
    p.add_argument('--wandb-project', default=None)
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-tags', default=None)
    run_multi_seed(p.parse_args())


if __name__ == '__main__':
    main()
