"""Evaluate a checkpoint on test samples from its seen classes only."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.foal.datasets import load_cifar100, make_task_specs, IndexedImageDataset, cifar100_transforms
from src.foal.model import FOALModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    m = ckpt['model_config'].copy()
    m['device'] = args.device
    model = FOALModel(**m, pretrained=True)
    if model.projection is not None and ckpt.get('projection_state') is not None:
        model.projection.load_state_dict(ckpt['projection_state'])
    model.classifier.load_state_dict(ckpt['classifier_state'])
    model.eval()
    
    train, test = load_cifar100()
    order = ckpt['class_order']
    specs = make_task_specs(train['fine_label'], test['fine_label'], order,
                            ckpt['config']['dataset']['classes_per_task'])
    spec = specs[ckpt['task_number'] - 1]
    mapping = {old: new for new, old in enumerate(order)}
    transformed = cifar100_transforms(size=ckpt['config']['dataset']['image_size'],
                                     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                     interpolation='bicubic')
    test_ds = IndexedImageDataset(test, spec.seen_test_indices, mapping, transformed)
    loader = DataLoader(test_ds, batch_size=ckpt['config']['training']['batch_size'], shuffle=False)
    num_correct = num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(args.device)).argmax(dim=1).cpu()
            num_correct += int((pred == y).sum())
            num_samples += len(y)
    result = {'checkpoint': str(Path(args.checkpoint)), 'task': ckpt['task_number'],
              'accuracy': num_correct / num_samples if num_samples else 0.0,
              'evaluated_samples': num_samples}
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    main()
