"""Train one-pass F-OAL on CIFAR-100 with comprehensive experiment tracking.

Usage:
    python -m scripts.train --config configs/cifar100.yaml
    python -m scripts.train --config configs/cifar100.yaml --max-tasks 2 --limit-train-per-task 100
    python -m scripts.train --config configs/cifar100.yaml --wandb --wandb-offline
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.foal.datasets import (load_cifar100, make_class_order, make_task_specs,
                              IndexedImageDataset, cifar100_transforms, limit_indices_per_task)
from src.foal.model import FOALModel
from src.foal.metrics import summary_from_task_matrix, format_summary_percentages
from src.foal.environment import get_environment_info


def device_from_config(value: str) -> str:
    if value != 'auto':
        return value
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_task(model: FOALModel, loader: DataLoader, device: str) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            scores = model(x.to(device))
            correct += int((scores.argmax(dim=1).cpu() == y).sum())
            total += len(y)
    return correct / total if total else float('nan')


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    
    # Granular seed management (A12)
    base_seed = int(args.seed if args.seed is not None else cfg.get('seed', 42))
    class_order_seed = int(args.class_order_seed if args.class_order_seed is not None
                           else cfg.get('class_order_seed', base_seed))
    projection_seed = int(args.projection_seed if args.projection_seed is not None
                          else cfg.get('projection_seed', base_seed))
    stream_seed = int(args.stream_seed if args.stream_seed is not None
                      else cfg.get('stream_seed', base_seed))

    # Seed random generators
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    device = args.device or device_from_config(cfg['training'].get('device', 'auto'))
    data_cfg = cfg['dataset']
    model_cfg = cfg['model']

    train_source, test_source = load_cifar100()
    labels_train = train_source['fine_label']
    labels_test = test_source['fine_label']

    order = make_class_order(int(data_cfg['num_classes']), class_order_seed, data_cfg.get('class_order'))
    specs = make_task_specs(labels_train, labels_test, order, int(data_cfg['classes_per_task']))
    
    max_tasks = args.max_tasks or len(specs)
    if not (1 <= max_tasks <= len(specs)):
        raise ValueError(f'max-tasks must be between 1 and {len(specs)}')

    mapping = {original: i for i, original in enumerate(order)}
    trans = cifar100_transforms(size=int(data_cfg.get('image_size', 224)),
                               mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                               interpolation='bicubic')

    use_fusion = bool(model_cfg['feature_fusion']) and not args.no_feature_fusion
    use_projection = bool(model_cfg['smooth_projection']) and not args.no_projection
    rls_dtype = model_cfg.get('rls_dtype', 'float64')

    # Experiment identity and method classification
    if use_fusion and use_projection:
        method = 'foal'
    elif not use_fusion and use_projection:
        method = 'foal_no_fusion'
    elif use_fusion and not use_projection:
        method = 'foal_no_projection'
    else:
        method = 'foal_neither'

    # Determine experiment type
    if args.experiment_type:
        experiment_type = args.experiment_type
    elif max_tasks < len(specs) or args.limit_train_per_task or args.limit_test_per_task:
        experiment_type = 'pilot'
    elif not use_fusion or not use_projection:
        experiment_type = 'ablation'
    else:
        experiment_type = 'reproduction'

    model = FOALModel(backbone_name=model_cfg['backbone'],
                      pretrained=bool(model_cfg['pretrained']),
                      feature_fusion=use_fusion,
                      smooth_projection=use_projection,
                      projection_dim=int(model_cfg['projection_dim']),
                      projection_seed=projection_seed,
                      gamma=float(model_cfg['gamma']),
                      rls_dtype=rls_dtype,
                      device=device)
    model.eval()

    run_name = args.run_name or f'{method}_seed{base_seed}_{experiment_type}'
    outdir = Path(cfg['logging'].get('results_dir', 'results')) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Capture complete environment and provenance (A16)
    env_info = get_environment_info()

    metadata = {
        'paper': 'F-OAL NeurIPS 2024',
        'method': method,
        'experiment_type': experiment_type,
        'run_name': run_name,
        'seeds': {
            'base': base_seed,
            'class_order': class_order_seed,
            'projection': projection_seed,
            'stream': stream_seed,
        },
        'model_config': {
            'backbone': model_cfg['backbone'],
            'pretrained': bool(model_cfg['pretrained']),
            'feature_fusion': use_fusion,
            'smooth_projection': use_projection,
            'projection_dim': int(model_cfg['projection_dim']),
            'gamma': float(model_cfg['gamma']),
            'rls_dtype': rls_dtype,
        },
        'training_protocol': {
            'dataset': data_cfg['name'],
            'num_classes': int(data_cfg['num_classes']),
            'classes_per_task': int(data_cfg['classes_per_task']),
            'max_tasks': max_tasks,
            'limit_train_per_task': args.limit_train_per_task,
            'limit_test_per_task': args.limit_test_per_task,
            'batch_size': int(cfg['training']['batch_size']),
            'device': device,
            'one_pass': True,
        },
        'class_order': order,
        'environment': env_info,
        'config': cfg,
    }
    (outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # Weights & Biases experiment tracking (A13)
    wandb_run = None
    use_wandb = args.wandb or cfg['logging'].get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            if args.wandb_offline:
                os.environ['WANDB_MODE'] = 'offline'

            project = args.wandb_project or cfg['logging'].get('project', 'hcmus-foal-reproduction')
            entity = args.wandb_entity or cfg['logging'].get('entity')
            group = args.wandb_group or f'{method}_{experiment_type}'
            tags = [experiment_type, method, f'seed_{base_seed}']
            if args.wandb_tags:
                tags.extend([t.strip() for t in args.wandb_tags.split(',') if t.strip()])

            wandb_run = wandb.init(project=project, entity=entity, group=group,
                                   name=run_name, config=metadata, tags=tags)
        except Exception as exc:
            print(f'Warning: Weights & Biases initialization failed: {exc}. Continuing training without W&B.')
            wandb_run = None

    matrix: list[list[float]] = []
    seen_specs = []
    start_time = time.monotonic()

    try:
        for spec in specs[:max_tasks]:
            model.classifier.expand_classes(spec.total_seen_classes)
            
            chosen_train_ids = limit_indices_per_task(
                spec.train_indices, args.limit_train_per_task, stream_seed + spec.task_id
            )
            training_ds = IndexedImageDataset(train_source, chosen_train_ids, mapping, trans)
            gen = torch.Generator().manual_seed(stream_seed + spec.task_id)
            loader = DataLoader(training_ds, batch_size=int(cfg['training']['batch_size']),
                                shuffle=True, generator=gen, drop_last=False,
                                num_workers=int(cfg['training']['num_workers']))
            
            for x, y in loader:
                model.partial_fit(x.to(device), y.to(device))
            
            seen_specs.append(spec)
            per_task: list[float] = []

            for previous_spec in seen_specs:
                task_test_ids = [idx for idx in previous_spec.seen_test_indices
                                 if int(labels_test[idx]) in set(previous_spec.new_original_labels)]
                if args.limit_test_per_task is not None:
                    task_test_ids = limit_indices_per_task(
                        task_test_ids, args.limit_test_per_task, stream_seed + previous_spec.task_id
                    )
                test_ds = IndexedImageDataset(test_source, task_test_ids, mapping, trans)
                test_loader = DataLoader(test_ds, batch_size=int(cfg['training']['batch_size']),
                                         shuffle=False, num_workers=int(cfg['training']['num_workers']))
                per_task.append(evaluate_task(model, test_loader, device))

            matrix.append(per_task)
            stage_acc = sum(per_task) / len(per_task)
            elapsed = round(time.monotonic() - start_time, 2)
            
            record = {
                'task': spec.task_id + 1,
                'task_accuracy': per_task,
                'average_accuracy': stage_acc,
                'seen_classes': spec.total_seen_classes,
                'samples_seen': model.classifier.samples_seen,
                'duration_seconds': elapsed,
            }
            print(json.dumps(record))

            if wandb_run:
                try:
                    step_metrics = {
                        'task': spec.task_id + 1,
                        'stage_accuracy': stage_acc,
                        'stage_accuracy_pct': stage_acc * 100,
                        'samples_seen': model.classifier.samples_seen,
                        'duration_seconds': elapsed,
                    }
                    for i, acc_val in enumerate(per_task):
                        step_metrics[f'task_acc/task_{i+1}'] = acc_val
                        step_metrics[f'task_acc_pct/task_{i+1}'] = acc_val * 100
                    wandb_run.log(step_metrics)
                except Exception as exc:
                    print(f'Warning: W&B step log error: {exc}')

            # Save checkpoint with complete reproducible configuration
            checkpoint_path = outdir / f'task_{spec.task_id+1:02d}.pt'
            torch.save({
                'model_config': {
                    'backbone_name': model_cfg['backbone'],
                    'feature_fusion': use_fusion,
                    'smooth_projection': use_projection,
                    'projection_dim': int(model_cfg['projection_dim']),
                    'projection_seed': projection_seed,
                    'gamma': float(model_cfg['gamma']),
                    'rls_dtype': rls_dtype,
                },
                'projection_state': model.projection.state_dict() if model.projection else None,
                'classifier_state': model.classifier.state_dict(),
                'class_order': order,
                'task_number': spec.task_id + 1,
                'metadata': metadata,
                'config': cfg,
            }, checkpoint_path)

        # Compute formal metrics
        metrics = summary_from_task_matrix(matrix)
        total_duration = round(time.monotonic() - start_time, 2)
        metrics['total_duration_seconds'] = total_duration

        # Save metrics and summary
        metrics_payload = {'task_matrix': matrix, **metrics}
        (outdir / 'metrics.json').write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')

        with (outdir / 'task_accuracy.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['stage', 'task', 'accuracy'])
            for stage, row in enumerate(matrix, start=1):
                for i, val in enumerate(row, start=1):
                    writer.writerow([stage, i, val])

        formatted_pcts = format_summary_percentages(metrics)
        print('SUMMARY (ratios):', json.dumps(metrics))
        print(f"SUMMARY (percentage): Avg Incr Acc: {formatted_pcts.get('average_incremental_accuracy')}, "
              f"Last Acc: {formatted_pcts.get('last_accuracy')}, Forgetting: {formatted_pcts.get('forgetting')}")

        if wandb_run:
            try:
                wandb_run.log({
                    'summary/average_incremental_accuracy': metrics['average_incremental_accuracy'],
                    'summary/average_incremental_accuracy_pct': metrics['average_incremental_accuracy'] * 100,
                    'summary/last_accuracy': metrics['last_accuracy'],
                    'summary/last_accuracy_pct': metrics['last_accuracy'] * 100,
                    'summary/forgetting': metrics['forgetting'],
                    'summary/forgetting_pct': metrics['forgetting'] * 100,
                    'summary/total_duration_seconds': total_duration,
                })
                # Upload evaluation artifacts
                artifact = wandb.Artifact(name=f'{run_name}_results', type='evaluation',
                                          description=f'Evaluation results for {run_name}')
                artifact.add_file(str(outdir / 'metrics.json'))
                artifact.add_file(str(outdir / 'metadata.json'))
                artifact.add_file(str(outdir / 'task_accuracy.csv'))
                wandb_run.log_artifact(artifact)
            except Exception as exc:
                print(f'Warning: W&B summary/artifact error: {exc}')

        return metrics
    finally:
        if wandb_run:
            try:
                wandb_run.finish()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='Train one-pass F-OAL on CIFAR-100.')
    parser.add_argument('--config', default='configs/cifar100.yaml')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Pilot only: limit number of sequential tasks')
    parser.add_argument('--limit-train-per-task', type=int, default=None,
                        help='Pilot only: limit training samples per task')
    parser.add_argument('--limit-test-per-task', type=int, default=None,
                        help='Pilot only: limit test samples evaluated per task')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (overrides config seed)')
    parser.add_argument('--class-order-seed', type=int, default=None,
                        help='Dedicated seed for dataset class permutation')
    parser.add_argument('--projection-seed', type=int, default=None,
                        help='Dedicated seed for random projection matrix')
    parser.add_argument('--stream-seed', type=int, default=None,
                        help='Dedicated seed for data stream ordering and sampling')
    parser.add_argument('--device', default=None,
                        help="Device to use ('cpu', 'cuda', etc.)")
    parser.add_argument('--experiment-type', default=None,
                        choices=['pilot', 'reproduction', 'ablation', 'additional-analysis'],
                        help='Classification of the experiment')
    parser.add_argument('--no-feature-fusion', action='store_true',
                        help='Ablation: disable feature fusion')
    parser.add_argument('--no-projection', action='store_true',
                        help='Ablation: disable smooth projection')
    parser.add_argument('--run-name', default=None,
                        help='Custom name for this experiment run')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases experiment logging')
    parser.add_argument('--wandb-offline', action='store_true',
                        help='Log to Weights & Biases in offline mode')
    parser.add_argument('--wandb-project', default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', default=None,
                        help='Weights & Biases entity/username')
    parser.add_argument('--wandb-group', default=None,
                        help='Weights & Biases experiment group')
    parser.add_argument('--wandb-tags', default=None,
                        help='Comma-separated tags for Weights & Biases')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
