"""Automatic experiment result aggregation and summarization.

Scans the results directory for completed runs (containing metadata.json and metrics.json),
aggregates runs by method and experiment type, computes mean +/- std across seeds,
and exports structured CSV tables.

Usage:
    python -m scripts.summarize_runs --results-dir results
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_run(run_dir: Path) -> dict[str, Any] | None:
    """Read and validate a single run's metadata.json and metrics.json."""
    meta_path = run_dir / 'metadata.json'
    metrics_path = run_dir / 'metrics.json'
    if not (meta_path.exists() and metrics_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    # Handle both old and new metadata schemas
    method = meta.get('method')
    if not method:
        ff = meta.get('feature_fusion', meta.get('model_config', {}).get('feature_fusion', True))
        sp = meta.get('smooth_projection', meta.get('model_config', {}).get('smooth_projection', True))
        if ff and sp: method = 'foal'
        elif not ff and sp: method = 'foal_no_fusion'
        elif ff and not sp: method = 'foal_no_projection'
        else: method = 'foal_neither'

    exp_type = meta.get('experiment_type', meta.get('experimental_status', 'UNKNOWN')).upper()
    seed = meta.get('seeds', {}).get('base', meta.get('config', {}).get('seed'))
    device = meta.get('device', meta.get('training_protocol', {}).get('device', 'unknown'))
    git_commit = meta.get('git_commit', meta.get('environment', {}).get('git', {}).get('commit'))
    if git_commit and len(git_commit) > 7:
        git_commit = git_commit[:7]

    num_tasks = metrics.get('num_tasks', len(metrics.get('task_matrix', [])))

    return {
        'run_name': run_dir.name,
        'run_dir': str(run_dir),
        'method': method,
        'experiment_type': exp_type,
        'seed': seed,
        'num_tasks': num_tasks,
        'average_incremental_accuracy': metrics.get('average_incremental_accuracy'),
        'last_accuracy': metrics.get('last_accuracy'),
        'forgetting': metrics.get('forgetting'),
        'total_duration_seconds': metrics.get('total_duration_seconds'),
        'git_commit': git_commit,
        'device': device,
    }


def compute_group_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate mean and standard deviation for a group of runs."""
    n = len(runs)
    stats: dict[str, Any] = {'count': n}
    for metric_name in ['average_incremental_accuracy', 'last_accuracy', 'forgetting']:
        vals = [r[metric_name] for r in runs if r.get(metric_name) is not None]
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        if len(vals) > 1:
            var_v = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
            std_v = math.sqrt(var_v)
        else:
            std_v = 0.0
        stats[f'{metric_name}_mean'] = mean_v
        stats[f'{metric_name}_std'] = std_v
        stats[f'{metric_name}_display'] = f'{mean_v * 100:.2f} +/- {std_v * 100:.2f}%'
    return stats


def summarize_results(results_dir: str = 'results', output_dir: str | None = None,
                      filter_type: str | None = None) -> dict[str, Any]:
    base_path = Path(results_dir)
    out_path = Path(output_dir or results_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        print(f'Results directory does not exist: {base_path}')
        return {'individual_runs': [], 'groups': {}}

    all_runs: list[dict[str, Any]] = []
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            run_data = load_run(item)
            if run_data is not None:
                if filter_type and run_data['experiment_type'] != filter_type:
                    continue
                all_runs.append(run_data)

    if not all_runs:
        print(f'No completed runs found under {base_path}')
        return {'individual_runs': [], 'groups': {}}

    # Write individual runs CSV
    individual_csv = out_path / 'summary_runs.csv'
    with individual_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'method', 'experiment_type', 'seed', 'num_tasks',
            'avg_inc_acc', 'last_acc', 'forgetting', 'duration_s', 'git_commit', 'device'
        ])
        for r in all_runs:
            writer.writerow([
                r['run_name'], r['method'], r['experiment_type'], r['seed'], r['num_tasks'],
                f"{r['average_incremental_accuracy']:.4f}" if r['average_incremental_accuracy'] is not None else '',
                f"{r['last_accuracy']:.4f}" if r['last_accuracy'] is not None else '',
                f"{r['forgetting']:.4f}" if r['forgetting'] is not None else '',
                r['total_duration_seconds'] or '',
                r['git_commit'] or '',
                r['device'] or ''
            ])

    # Group by (method, experiment_type)
    groups = defaultdict(list)
    for r in all_runs:
        group_key = (r['method'], r['experiment_type'])
        groups[group_key].append(r)

    # Write aggregated summary CSV
    aggregated_csv = out_path / 'summary_aggregated.csv'
    with aggregated_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'experiment_type', 'num_runs', 'avg_inc_acc_mean_std', 'last_acc_mean_std', 'forgetting_mean_std'])
        for (m, exp), run_list in sorted(groups.items()):
            stats = compute_group_stats(run_list)
            writer.writerow([
                m, exp, stats['count'],
                stats.get('average_incremental_accuracy_display', 'N/A'),
                stats.get('last_accuracy_display', 'N/A'),
                stats.get('forgetting_display', 'N/A')
            ])

    # Check for ablation runs and export specialized summary if present
    ablation_runs = [r for r in all_runs if 'ablation' in r['run_name'] or r['experiment_type'] == 'ablation']
    if ablation_runs:
        ablation_csv = out_path / 'summary_ablation.csv'
        with ablation_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['run_name', 'method', 'seed', 'avg_inc_acc', 'last_acc', 'forgetting'])
            for r in ablation_runs:
                writer.writerow([
                    r['run_name'], r['method'], r['seed'],
                    f"{r['average_incremental_accuracy'] * 100:.2f}%" if r['average_incremental_accuracy'] is not None else '',
                    f"{r['last_accuracy'] * 100:.2f}%" if r['last_accuracy'] is not None else '',
                    f"{r['forgetting'] * 100:.2f}%" if r['forgetting'] is not None else ''
                ])

    # Print summary table
    print('\n' + '=' * 85)
    print(f'EXPERIMENT SUMMARY REPORT ({len(all_runs)} runs found)')
    print('=' * 85)
    print(f"{'Method':<20} {'Type':<15} {'Runs':<6} {'Avg Incr Acc':<22} {'Last Acc':<20}")
    print('-' * 85)
    for (m, exp), run_list in sorted(groups.items()):
        stats = compute_group_stats(run_list)
        print(f"{m:<20} {exp:<15} {stats['count']:<6} "
              f"{stats.get('average_incremental_accuracy_display', 'N/A'):<22} "
              f"{stats.get('last_accuracy_display', 'N/A'):<20}")
    print('=' * 85)
    print(f'Saved run details to:  {individual_csv}')
    print(f'Saved aggregation to:  {aggregated_csv}\n')

    return {'individual_runs': all_runs, 'groups': {f'{k[0]}@{k[1]}': compute_group_stats(v) for k, v in groups.items()}}


def main():
    p = argparse.ArgumentParser(description='Aggregate and summarize F-OAL experiments.')
    p.add_argument('--results-dir', default='results',
                   help='Directory containing experiment run folders')
    p.add_argument('--output-dir', default=None,
                   help='Directory to output summary CSV files')
    p.add_argument('--filter-type', default=None,
                   help='Filter runs by experiment type')
    args = p.parse_args()
    summarize_results(results_dir=args.results_dir, output_dir=args.output_dir,
                      filter_type=args.filter_type)


if __name__ == '__main__':
    main()
