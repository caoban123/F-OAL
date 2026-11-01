"""Evaluate a checkpoint on test samples from its seen classes only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.foal.datasets import (load_cifar100, make_task_specs, IndexedImageDataset,
                              cifar100_transforms, limit_indices_per_task)
from src.foal.model import FOALModel


def evaluate_checkpoint(checkpoint_path: str, device: str = 'cpu',
                        limit_per_task: int | None = None) -> dict:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    m = ckpt['model_config'].copy()
    m['device'] = device
    model = FOALModel(**m, pretrained=True)
    if model.projection is not None and ckpt.get('projection_state') is not None:
        model.projection.load_state_dict(ckpt['projection_state'])
    model.classifier.load_state_dict(ckpt['classifier_state'])
    model.eval()
    
    train, test = load_cifar100()
    order = ckpt['class_order']
    specs = make_task_specs(train['fine_label'], test['fine_label'], order,
                            ckpt['config']['dataset']['classes_per_task'])
    task_number = ckpt['task_number']
    seen_specs = specs[:task_number]
    mapping = {old: new for new, old in enumerate(order)}
    transformed = cifar100_transforms(size=ckpt['config']['dataset']['image_size'],
                                     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                     interpolation='bicubic')
    
    per_task_accuracies = []
    task_details = {}
    all_scores = []
    total_correct = 0
    total_samples = 0
    
    labels_test = test['fine_label']
    seed = int(ckpt['config']['seed'])

    with torch.no_grad():
        for spec in seen_specs:
            task_test_ids = [idx for idx in spec.seen_test_indices
                             if int(labels_test[idx]) in set(spec.new_original_labels)]
            if limit_per_task is not None:
                task_test_ids = limit_indices_per_task(task_test_ids, limit_per_task, seed + spec.task_id)
            
            test_ds = IndexedImageDataset(test, task_test_ids, mapping, transformed)
            loader = DataLoader(test_ds, batch_size=ckpt['config']['training']['batch_size'], shuffle=False)
            task_correct = 0
            task_total = 0
            
            for x, y in loader:
                scores = model(x.to(device)).cpu()
                if len(all_scores) < 200:
                    all_scores.append(scores)
                task_correct += int((scores.argmax(dim=1) == y).sum())
                task_total += len(y)
                
            acc = task_correct / task_total if task_total else float('nan')
            per_task_accuracies.append(acc)
            task_details[f'task_{spec.task_id+1}'] = {
                'samples': task_total,
                'correct': task_correct,
                'accuracy': acc
            }
            total_correct += task_correct
            total_samples += task_total

    concatenated_scores = torch.cat(all_scores, dim=0) if all_scores else torch.empty(0)
    scores_finite = bool(torch.isfinite(concatenated_scores).all().item()) if len(concatenated_scores) else True

    score_stats = {
        'min': float(concatenated_scores.min().item()) if len(concatenated_scores) else None,
        'max': float(concatenated_scores.max().item()) if len(concatenated_scores) else None,
        'mean': float(concatenated_scores.mean().item()) if len(concatenated_scores) else None,
        'std': float(concatenated_scores.std().item()) if len(concatenated_scores) else None,
        'all_finite': scores_finite
    }

    result = {
        'checkpoint': str(Path(checkpoint_path)),
        'task': task_number,
        'accuracy': total_correct / total_samples if total_samples else 0.0,
        'per_task_accuracy': per_task_accuracies,
        'task_details': task_details,
        'evaluated_samples': total_samples,
        'score_stats': score_stats
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit-per-task', type=int, default=None)
    args = p.parse_args()
    evaluate_checkpoint(args.checkpoint, device=args.device, limit_per_task=args.limit_per_task)


if __name__ == '__main__':
    main()
