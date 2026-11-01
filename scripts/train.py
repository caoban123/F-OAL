"""Train one-pass F-OAL on CIFAR-100 and log per-task evaluation.

Usage: python -m scripts.train --config configs/cifar100.yaml --max-tasks 2
The --max-tasks and --limit-train-per-task flags are for PILOT runs only.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import time
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from src.foal.datasets import (load_cifar100, make_class_order, make_task_specs,
    IndexedImageDataset, cifar100_transforms, limit_indices_per_task)
from src.foal.model import FOALModel
from src.foal.metrics import summary_from_task_matrix


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


def run(args: argparse.Namespace):
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seed = int(cfg['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device_from_config(cfg['training']['device'])
    train_source, test_source = load_cifar100()
    labels_train = train_source['fine_label']
    labels_test = test_source['fine_label']
    data_cfg = cfg['dataset']; model_cfg = cfg['model']
    order = make_class_order(int(data_cfg['num_classes']), seed, data_cfg.get('class_order'))
    specs = make_task_specs(labels_train, labels_test, order, int(data_cfg['classes_per_task']))
    max_tasks = args.max_tasks or len(specs)
    if not 1 <= max_tasks <= len(specs):
        raise ValueError('max-tasks is out of range')
    mapping = {original: i for i, original in enumerate(order)}
    
    # ViT transforms: (0.5, 0.5, 0.5) normalization and bicubic interpolation
    trans = cifar100_transforms(size=int(data_cfg['image_size']),
                               mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                               interpolation='bicubic')
    
    use_fusion = bool(model_cfg['feature_fusion']) and not args.no_feature_fusion
    use_projection = bool(model_cfg['smooth_projection']) and not args.no_projection
    rls_dtype = model_cfg.get('rls_dtype', 'float64')

    model = FOALModel(backbone_name=model_cfg['backbone'],
                      pretrained=bool(model_cfg['pretrained']),
                      feature_fusion=use_fusion,
                      smooth_projection=use_projection,
                      projection_dim=int(model_cfg['projection_dim']),
                      projection_seed=seed,
                      gamma=float(model_cfg['gamma']),
                      rls_dtype=rls_dtype,
                      device=device)
    model.eval()

    run_name = args.run_name or f'foal_seed{seed}_{"noff" if not use_fusion else "ff"}_{"nosp" if not use_projection else "sp"}'
    outdir = Path(cfg['logging']['results_dir']) / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None
    metadata = dict(git_commit=git_commit, paper='F-OAL NeurIPS 2024', config=cfg, class_order=order,
                    feature_fusion=use_fusion, smooth_projection=use_projection,
                    max_tasks=max_tasks, limit_train_per_task=args.limit_train_per_task,
                    limit_test_per_task=args.limit_test_per_task,
                    device=device, experimental_status='PILOT' if max_tasks < len(specs) or args.limit_train_per_task or args.limit_test_per_task else 'FULL_RUN')
    (outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    wandb_run = None
    if args.wandb or cfg['logging'].get('use_wandb', False):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('pip install wandb or disable --wandb') from exc
        wandb_run = wandb.init(project=cfg['logging']['project'],
                               entity=cfg['logging'].get('entity'), name=run_name,
                               config=metadata, tags=['pilot' if metadata['experimental_status']=='PILOT' else 'reproduction'])

    matrix = []
    seen_specs = []
    start_time = time.monotonic()
    try:
        for spec in specs[:max_tasks]:
            model.classifier.expand_classes(spec.total_seen_classes)
            chosen = limit_indices_per_task(spec.train_indices, args.limit_train_per_task, seed + spec.task_id)
            training = IndexedImageDataset(train_source, chosen, mapping, trans)
            gen = torch.Generator().manual_seed(seed + spec.task_id)
            loader = DataLoader(training, batch_size=int(cfg['training']['batch_size']),
                                shuffle=True, generator=gen, drop_last=False,
                                num_workers=int(cfg['training']['num_workers']))
            for x, y in loader:
                model.partial_fit(x.to(device), y.to(device))
            seen_specs.append(spec)
            per_task = []
            for previous_spec in seen_specs:
                task_test_ids = [idx for idx in previous_spec.seen_test_indices
                                 if int(labels_test[idx]) in set(previous_spec.new_original_labels)]
                if args.limit_test_per_task is not None:
                    task_test_ids = limit_indices_per_task(task_test_ids, args.limit_test_per_task, seed + previous_spec.task_id)
                test_ds = IndexedImageDataset(test_source, task_test_ids, mapping, trans)
                test_loader = DataLoader(test_ds, batch_size=int(cfg['training']['batch_size']),
                                         shuffle=False, num_workers=int(cfg['training']['num_workers']))
                per_task.append(evaluate_task(model, test_loader, device))
            matrix.append(per_task)
            acc = sum(per_task) / len(per_task)
            record = dict(task=spec.task_id+1, task_accuracy=per_task,
                          average_accuracy=acc, seen_classes=spec.total_seen_classes,
                          samples_seen=model.classifier.samples_seen,
                          duration_seconds=round(time.monotonic()-start_time, 2))
            print(json.dumps(record))
            if wandb_run:
                wandb_run.log({f'task_acc/{i+1}': v for i,v in enumerate(per_task)} |
                              {'task': spec.task_id+1, 'avg_seen_accuracy': acc,
                               'samples_seen': model.classifier.samples_seen})
            # Checkpoint contains full configuration to reproduce evaluation exactly
            torch.save(dict(model_config=dict(backbone_name=model_cfg['backbone'],
                                              feature_fusion=use_fusion,
                                              smooth_projection=use_projection,
                                              projection_dim=int(model_cfg['projection_dim']),
                                              projection_seed=seed,
                                              gamma=float(model_cfg['gamma']),
                                              rls_dtype=rls_dtype),
                            projection_state=model.projection.state_dict() if model.projection else None,
                            classifier_state=model.classifier.state_dict(),
                            class_order=order,
                            task_number=spec.task_id+1,
                            config=cfg), outdir / f'task_{spec.task_id+1:02d}.pt')
        metrics = summary_from_task_matrix(matrix)
        (outdir / 'metrics.json').write_text(json.dumps(dict(task_matrix=matrix, **metrics), indent=2), encoding='utf-8')
        with (outdir / 'task_accuracy.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['stage', 'task', 'accuracy'])
            for stage, row in enumerate(matrix, start=1):
                for i, val in enumerate(row, start=1):
                    writer.writerow([stage, i, val])
        print('SUMMARY:', json.dumps(metrics))
        if wandb_run:
            wandb_run.log({'summary/average_incremental_accuracy': metrics['average_incremental_accuracy'],
                           'summary/last_accuracy': metrics['last_accuracy'],
                           'summary/forgetting_proxy': metrics['forgetting']})
        return metrics
    finally:
        if wandb_run:
            wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cifar100.yaml')
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--limit-train-per-task', type=int, default=None)
    parser.add_argument('--limit-test-per-task', type=int, default=None)
    parser.add_argument('--no-feature-fusion', action='store_true')
    parser.add_argument('--no-projection', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--run-name', default=None)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
