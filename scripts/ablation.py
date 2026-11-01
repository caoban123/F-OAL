"""Execute 2x2 feature fusion / smooth projection variants."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/cifar100.yaml')
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--limit-train-per-task', type=int)
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    for ff, sp, name in [(True, True, 'full'), (False, True, 'no_fusion'),
                         (True, False, 'no_projection'), (False, False, 'neither')]:
        command = [sys.executable, '-m', 'scripts.train', '--config', args.config,
                   '--run-name', 'ablation_' + name]
        if not ff: command.append('--no-feature-fusion')
        if not sp: command.append('--no-projection')
        if args.max_tasks: command.extend(['--max-tasks', str(args.max_tasks)])
        if args.limit_train_per_task:
            command.extend(['--limit-train-per-task', str(args.limit_train_per_task)])
        if args.wandb: command.append('--wandb')
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
