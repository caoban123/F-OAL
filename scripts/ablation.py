"""Execute 2x2 Feature Fusion / Smooth Projection ablation study.

Variants:
1. full:          Feature Fusion = ON,  Smooth Projection = ON  (Full F-OAL)
2. no_fusion:     Feature Fusion = OFF, Smooth Projection = ON
3. no_projection: Feature Fusion = ON,  Smooth Projection = OFF
4. neither:       Feature Fusion = OFF, Smooth Projection = OFF

Usage:
    python -m scripts.ablation --config configs/cifar100.yaml
    python -m scripts.ablation --max-tasks 2 --limit-train-per-task 100 --limit-test-per-task 100
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from scripts.summarize_runs import summarize_results


def main():
    p = argparse.ArgumentParser(description='Run 2x2 F-OAL ablation experiments.')
    p.add_argument('--config', default='configs/cifar100.yaml')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--max-tasks', type=int, default=None)
    p.add_argument('--limit-train-per-task', type=int, default=None)
    p.add_argument('--limit-test-per-task', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-offline', action='store_true')
    p.add_argument('--wandb-project', default=None)
    args = p.parse_args()

    variants = [
        (True, True, 'full'),
        (False, True, 'no_fusion'),
        (True, False, 'no_projection'),
        (False, False, 'neither'),
    ]

    seed_suffix = f'_seed{args.seed}' if args.seed is not None else ''

    for ff, sp, name in variants:
        run_name = f'ablation_{name}{seed_suffix}'
        command = [
            sys.executable, '-m', 'scripts.train',
            '--config', args.config,
            '--run-name', run_name,
            '--experiment-type', 'ablation',
        ]
        if not ff:
            command.append('--no-feature-fusion')
        if not sp:
            command.append('--no-projection')
        if args.seed is not None:
            command.extend(['--seed', str(args.seed)])
        if args.max_tasks is not None:
            command.extend(['--max-tasks', str(args.max_tasks)])
        if args.limit_train_per_task is not None:
            command.extend(['--limit-train-per-task', str(args.limit_train_per_task)])
        if args.limit_test_per_task is not None:
            command.extend(['--limit-test-per-task', str(args.limit_test_per_task)])
        if args.device is not None:
            command.extend(['--device', args.device])
        if args.wandb:
            command.append('--wandb')
        if args.wandb_offline:
            command.append('--wandb-offline')
        if args.wandb_project:
            command.extend(['--wandb-project', args.wandb_project])

        print(f"\n{'='*70}\nExecuting ablation variant: {name} ({run_name})\n{'='*70}", flush=True)
        subprocess.run(command, check=True)

    print('\nAll 4 ablation variants completed. Summarizing ablation results...')
    summarize_results(results_dir='results')


if __name__ == '__main__':
    main()
