"""Automatic experiment result aggregation and summarization.

Scans the results directory for runs, validates completeness (including final task checkpoint),
and aggregates runs strictly by matching protocol to prevent mixing pilot, full reproduction,
controlled projection initializations, or incomplete runs.

Usage:
    python -m scripts.summarize_runs --results-dir results
"""
from __future__ import annotations
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.foal.metrics import forgetting_official_code, forgetting_past_tasks


def load_run(run_dir: Path) -> dict[str, Any] | None:
    """Read and validate a single run's metadata.json and metrics.json.

    Verifies that the run is complete, including the presence of the final checkpoint.
    """
    meta_path = run_dir / 'metadata.json'
    metrics_path = run_dir / 'metrics.json'
    if not (meta_path.exists() and metrics_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    # Handle method extraction
    method = meta.get('method')
    if not method:
        ff = meta.get('feature_fusion', meta.get('model_config', {}).get('feature_fusion', True))
        sp = meta.get('smooth_projection', meta.get('model_config', {}).get('smooth_projection', True))
        if ff and sp:
            method = 'foal'
        elif not ff and sp:
            method = 'foal_no_fusion'
        elif ff and not sp:
            method = 'foal_no_projection'
        else:
            method = 'foal_neither'

    exp_type = meta.get('experiment_type', meta.get('experimental_status', 'UNKNOWN')).upper()
    seed = meta.get('seeds', {}).get('base', meta.get('config', {}).get('seed'))
    device = meta.get('device', meta.get('training_protocol', {}).get('device', 'unknown'))
    git_commit = (
        meta.get('git_commit')
        or meta.get('official_commit_sha')
        or meta.get('environment', {}).get('git', {}).get('commit')
    )
    if git_commit and len(git_commit) > 7:
        git_commit = git_commit[:7]

    protocol = meta.get('training_protocol', {})
    model_cfg = meta.get('model_config', meta.get('config', {}).get('model', {}))
    proj_meta = meta.get('projection_metadata') or {}
    proj_cfg = meta.get('config', {}).get('projection', {}) or {}

    task_matrix = metrics.get('task_matrix', [])
    num_tasks = metrics.get('num_tasks', len(task_matrix))
    expected_tasks = protocol.get('max_tasks', num_tasks)

    limit_train = protocol.get('limit_train_per_task', meta.get('limit_train_per_task'))
    limit_test = protocol.get('limit_test_per_task', meta.get('limit_test_per_task'))
    backbone = model_cfg.get('backbone', 'unknown')
    projection_dim = model_cfg.get('projection_dim', proj_cfg.get('dimension', 1000))
    projection_init = (
        model_cfg.get('projection_init')
        or proj_meta.get('initialization')
        or proj_cfg.get('initialization')
        or 'gaussian_std1'
    )
    gamma = model_cfg.get('gamma', 1.0)
    rls_dtype = model_cfg.get('rls_dtype', 'float64')
    ff = model_cfg.get('feature_fusion', meta.get('feature_fusion', True))
    sp = model_cfg.get('smooth_projection', meta.get('smooth_projection', True))

    seeds = meta.get('seeds', {})
    base_seed = seeds.get('base', meta.get('config', {}).get('seed'))
    class_order_seed = seeds.get('class_order', base_seed)
    projection_seed = seeds.get('projection', base_seed)
    stream_seed = seeds.get('stream', base_seed)

    if class_order_seed == base_seed and projection_seed == base_seed and stream_seed == base_seed:
        seed_coupling = 'coupled'
    else:
        c_tag = 'base' if class_order_seed == base_seed else f'fixed_{class_order_seed}'
        p_tag = 'base' if projection_seed == base_seed else f'fixed_{projection_seed}'
        s_tag = 'base' if stream_seed == base_seed else f'fixed_{stream_seed}'
        seed_coupling = f'c:{c_tag}_p:{p_tag}_s:{s_tag}'

    class_order = meta.get('class_order')
    class_order_mode = 'custom_order' if (class_order is not None and class_order_seed is None) else 'seeded_order'

    # Completeness verification:
    # 1. Must contain all required metric fields
    has_metrics = (
        'average_incremental_accuracy' in metrics and
        'last_accuracy' in metrics and
        'forgetting' in metrics and
        len(task_matrix) == expected_tasks
    )

    # 2. Must contain final checkpoint file (for our implementation) or verified official CSV/log (for official_foal)
    final_ckpt = run_dir / f'task_{expected_tasks:02d}.pt'
    has_final_ckpt = final_ckpt.exists() and final_ckpt.stat().st_size > 0
    if not has_final_ckpt and (method == 'official_foal' or exp_type == 'OFFICIAL_REPRODUCTION'):
        official_csv = run_dir / 'cifar100_FOAL.csv'
        official_log = run_dir / 'official_stdout.log'
        has_final_ckpt = (
            official_csv.exists()
            and official_csv.stat().st_size > 0
            and official_log.exists()
            and 'Records have been written' in official_log.read_text(encoding='utf-8', errors='ignore')
        )

    is_complete = bool(has_metrics and has_final_ckpt)
    status = 'COMPLETE' if is_complete else 'INCOMPLETE'

    # Compute C0 dual forgetting definitions when complete
    f_past = None
    f_code = None
    if is_complete:
        f_past = metrics.get('forgetting_past_tasks', metrics.get('forgetting'))
        f_code = metrics.get('forgetting_official_code', metrics.get('official_forgetting_10_tasks'))
        if f_code is None and task_matrix:
            try:
                f_code = forgetting_official_code(task_matrix)
            except Exception:
                f_code = None

    # Strict protocol signature prevents accidental cross-protocol aggregation
    protocol_sig = (
        method,
        exp_type,
        bool(ff),
        bool(sp),
        int(expected_tasks),
        limit_train,
        limit_test,
        str(backbone),
        int(projection_dim),
        float(gamma),
        str(rls_dtype),
        str(seed_coupling),
        str(class_order_mode),
        str(projection_init),
    )

    return {
        'run_name': run_dir.name,
        'run_dir': str(run_dir),
        'method': method,
        'experiment_type': exp_type,
        'status': status,
        'is_complete': is_complete,
        'seed': seed,
        'num_tasks': expected_tasks,
        'limit_train_per_task': limit_train,
        'limit_test_per_task': limit_test,
        'backbone': backbone,
        'projection_dim': projection_dim,
        'projection_init': projection_init,
        'gamma': gamma,
        'protocol_sig': protocol_sig,
        'average_incremental_accuracy': metrics.get('average_incremental_accuracy') if is_complete else None,
        'last_accuracy': metrics.get('last_accuracy') if is_complete else None,
        'forgetting': metrics.get('forgetting') if is_complete else None,
        'forgetting_past_tasks': f_past,
        'forgetting_official_code': f_code,
        'total_duration_seconds': metrics.get('total_duration_seconds'),
        'git_commit': git_commit,
        'device': device,
    }


def compute_group_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate mean and standard deviation for a group of completed runs.

    When n == 1, sample standard deviation is undefined (None / 'N/A').
    """
    completed = [r for r in runs if r.get('is_complete')]
    n = len(completed)
    stats: dict[str, Any] = {'count': n, 'total_runs': len(runs), 'incomplete_count': len(runs) - n}
    for metric_name in [
        'average_incremental_accuracy',
        'last_accuracy',
        'forgetting',
        'forgetting_past_tasks',
        'forgetting_official_code',
    ]:
        vals = [r[metric_name] for r in completed if r.get(metric_name) is not None]
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        if len(vals) > 1:
            var_v = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
            std_v: float | None = math.sqrt(var_v)
            display_str = f'{mean_v * 100:.2f} +/- {std_v * 100:.2f}% (n={len(vals)})'
        else:
            std_v = None
            display_str = f'{mean_v * 100:.2f}% (n=1, std=N/A)'
        stats[f'{metric_name}_mean'] = mean_v
        stats[f'{metric_name}_std'] = std_v
        stats[f'{metric_name}_display'] = display_str
    return stats


def format_protocol_description(sig: tuple) -> str:
    """Format protocol signature into human-readable label."""
    if len(sig) == 14:
        method, exp_type, ff, sp, tasks, lim_train, lim_test, backbone, p_dim, gamma, dtype, seed_coup, order_mode, p_init = sig
    else:
        method, exp_type, ff, sp, tasks, lim_train, lim_test, backbone, p_dim, gamma, dtype, seed_coup, order_mode = sig
        p_init = 'gaussian_std1'
    data_desc = 'full data'
    if lim_train or lim_test:
        data_desc = f"train<={lim_train or 'all'}, test<={lim_test or 'all'}"
    desc = f"{method} | {exp_type} | {tasks} tasks ({data_desc}, D={p_dim}, gamma={gamma}, {dtype})"
    if not ff:
        desc += " [no_fusion]"
    if not sp:
        desc += " [no_proj]"
    elif p_init != 'gaussian_std1':
        desc += f" [proj:{p_init}]"
    if seed_coup != 'coupled':
        desc += f" [seeds: {seed_coup}]"
    if order_mode != 'seeded_order':
        desc += f" [{order_mode}]"
    return desc


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
                if filter_type and run_data['experiment_type'] != filter_type.upper():
                    continue
                all_runs.append(run_data)

    if not all_runs:
        print(f'No valid runs found under {base_path}')
        return {'individual_runs': [], 'groups': {}}

    # Write individual runs CSV
    individual_csv = out_path / 'summary_runs.csv'
    with individual_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'method', 'experiment_type', 'projection_init', 'status', 'seed', 'num_tasks',
            'limit_train', 'limit_test', 'avg_inc_acc', 'last_acc', 'forgetting',
            'forgetting_past_tasks', 'forgetting_official_code',
            'duration_s', 'git_commit', 'device'
        ])
        for r in all_runs:
            writer.writerow([
                r['run_name'], r['method'], r['experiment_type'], r.get('projection_init', 'gaussian_std1'),
                r['status'], r['seed'], r['num_tasks'],
                r['limit_train_per_task'] if r['limit_train_per_task'] is not None else 'full',
                r['limit_test_per_task'] if r['limit_test_per_task'] is not None else 'full',
                f"{r['average_incremental_accuracy']:.4f}" if r['average_incremental_accuracy'] is not None else 'N/A',
                f"{r['last_accuracy']:.4f}" if r['last_accuracy'] is not None else 'N/A',
                f"{r['forgetting']:.4f}" if r['forgetting'] is not None else 'N/A',
                f"{r['forgetting_past_tasks']:.4f}" if r.get('forgetting_past_tasks') is not None else 'N/A',
                f"{r['forgetting_official_code']:.4f}" if r.get('forgetting_official_code') is not None else 'N/A',
                r['total_duration_seconds'] or '',
                r['git_commit'] or '',
                r['device'] or ''
            ])

    # Group runs strictly by exact protocol signature
    groups = defaultdict(list)
    for r in all_runs:
        groups[r['protocol_sig']].append(r)

    def _sig_sort_key(item: tuple[tuple, list[dict[str, Any]]]) -> tuple:
        sig = item[0]
        return tuple(-1 if v is None else v for v in sig)

    # Write aggregated summary CSV
    aggregated_csv = out_path / 'summary_aggregated.csv'
    with aggregated_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'protocol_label', 'method', 'experiment_type', 'num_tasks', 'limit_train', 'limit_test',
            'completed_runs', 'total_runs', 'avg_inc_acc_mean_std', 'last_acc_mean_std',
            'forgetting_mean_std', 'forgetting_official_code_mean_std'
        ])
        for sig, run_list in sorted(groups.items(), key=_sig_sort_key):
            stats = compute_group_stats(run_list)
            label = format_protocol_description(sig)
            writer.writerow([
                label, sig[0], sig[1], sig[4],
                sig[5] if sig[5] is not None else 'full',
                sig[6] if sig[6] is not None else 'full',
                stats['count'], stats['total_runs'],
                stats.get('average_incremental_accuracy_display', 'N/A'),
                stats.get('last_accuracy_display', 'N/A'),
                stats.get('forgetting_display', 'N/A'),
                stats.get('forgetting_official_code_display', 'N/A'),
            ])

    # Specialized summary for ablation runs if present
    ablation_runs = [r for r in all_runs if 'ablation' in r['run_name'] or r['experiment_type'] == 'ABLATION']
    if ablation_runs:
        ablation_csv = out_path / 'summary_ablation.csv'
        with ablation_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['run_name', 'method', 'status', 'seed', 'avg_inc_acc', 'last_acc', 'forgetting'])
            for r in ablation_runs:
                writer.writerow([
                    r['run_name'], r['method'], r['status'], r['seed'],
                    f"{r['average_incremental_accuracy'] * 100:.2f}%" if r['average_incremental_accuracy'] is not None else 'N/A',
                    f"{r['last_accuracy'] * 100:.2f}%" if r['last_accuracy'] is not None else 'N/A',
                    f"{r['forgetting'] * 100:.2f}%" if r['forgetting'] is not None else 'N/A'
                ])

    # Print summary table
    print('\n' + '=' * 100)
    print(f'EXPERIMENT SUMMARY REPORT ({len(all_runs)} runs found, strictly grouped by protocol)')
    print('=' * 100)
    print(f"{'Protocol / Group':<55} {'Runs':<8} {'Avg Incr Acc':<20} {'Last Acc':<18}")
    print('-' * 100)
    for sig, run_list in sorted(groups.items(), key=_sig_sort_key):
        stats = compute_group_stats(run_list)
        label = format_protocol_description(sig)
        run_count_str = f"{stats['count']}/{stats['total_runs']}"
        print(f"{label:<55} {run_count_str:<8} "
              f"{stats.get('average_incremental_accuracy_display', 'N/A'):<20} "
              f"{stats.get('last_accuracy_display', 'N/A'):<18}")
    print('=' * 100)
    print(f'Saved run details to:  {individual_csv}')
    print(f'Saved aggregation to:  {aggregated_csv}\n')

    return {
        'individual_runs': all_runs,
        'groups': {format_protocol_description(k): compute_group_stats(v) for k, v in groups.items()}
    }


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
