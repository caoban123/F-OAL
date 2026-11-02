"""CIFAR-100 task stream from Hugging Face or torchvision; no training data replay.

Only use *training* samples from the active task. Evaluation can use test
samples from all seen classes; unseen test labels are never exposed.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Tuple, Sequence as SeqType, Any, Dict
import numpy as np
import torch


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    new_original_labels: tuple[int, ...]
    train_indices: tuple[int, ...]
    seen_test_indices: tuple[int, ...]
    total_seen_classes: int


def make_class_order(num_classes: int, seed: int = 42,
                     explicit: Sequence[int] | None = None) -> list[int]:
    if num_classes < 1:
        raise ValueError('num_classes must be positive')
    if explicit is not None:
        order = list(explicit)
        if len(order) != num_classes or sorted(order) != list(range(num_classes)):
            raise ValueError('Explicit order must be a permutation of all classes')
        return order
    return np.random.default_rng(seed).permutation(num_classes).tolist()


def make_task_specs(train_labels: Sequence[int], test_labels: Sequence[int],
                    order: Sequence[int], classes_per_task: int = 10) -> list[TaskSpec]:
    if classes_per_task < 1 or len(order) % classes_per_task:
        raise ValueError('classes_per_task must divide number of classes')
    if sorted(order) != list(range(len(order))):
        raise ValueError('Class order must be a permutation of 0..C-1')
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    if train.ndim != 1 or test.ndim != 1:
        raise ValueError('Labels must be 1D')
    specs = []
    seen = set()
    all_train_indices = set()
    for task_id, start in enumerate(range(0, len(order), classes_per_task)):
        new_labels = tuple(order[start:start+classes_per_task])
        new_set = set(new_labels)
        train_ids = np.flatnonzero(np.isin(train, list(new_set))).tolist()
        if any(i in all_train_indices for i in train_ids):
            raise AssertionError('A training sample would be replayed')
        all_train_indices.update(train_ids)
        seen.update(new_set)
        test_ids = np.flatnonzero(np.isin(test, list(seen))).tolist()
        specs.append(TaskSpec(task_id=task_id, new_original_labels=new_labels,
                              train_indices=tuple(train_ids), seen_test_indices=tuple(test_ids),
                              total_seen_classes=len(seen)))
    return specs


def load_cifar100(root: str = './data'):
    """Download and load CIFAR-100 dataset.
    
    Primary source is Hugging Face Datasets ('uoft-cs/cifar100').
    Falls back gracefully to torchvision.datasets.CIFAR100 if unavailable.
    """
    try:
        from datasets import load_dataset
        data = load_dataset('uoft-cs/cifar100')
        return data['train'], data['test']
    except Exception as exc:
        # Fallback to torchvision
        from torchvision.datasets import CIFAR100
        tv_train = CIFAR100(root=root, train=True, download=True)
        tv_test = CIFAR100(root=root, train=False, download=True)
        
        class TorchvisionCIFAR100Adapter:
            def __init__(self, tv_dataset):
                self.dataset = tv_dataset
                self.fine_label = list(tv_dataset.targets)
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                img, target = self.dataset[idx]
                return {'img': img, 'fine_label': target}
        
        return TorchvisionCIFAR100Adapter(tv_train), TorchvisionCIFAR100Adapter(tv_test)


def cifar100_transforms(size: int = 224,
                       mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                       std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                       interpolation: str = 'bicubic'):
    """Preprocessing transforms compatible with ViT backbone.
    
    Default normalization is (0.5, 0.5, 0.5) matching pre-trained Vision Transformers
    (such as timm vit_base_patch16_224).
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    
    interp_mode = InterpolationMode.BICUBIC if interpolation.lower() == 'bicubic' else InterpolationMode.BILINEAR
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=interp_mode),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class IndexedImageDataset:
    """A lazy dataset over selected sample indices and a fixed label mapping."""
    def __init__(self, source, indices: Sequence[int], label_to_incremental: dict[int, int],
                 transform):
        self.source = source
        self.indices = list(indices)
        self.label_to_incremental = label_to_incremental
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = self.source[self.indices[index]]
        image = sample['img'].convert('RGB')
        label = int(sample['fine_label'])
        if label not in self.label_to_incremental:
            raise ValueError('Encountered unseen label in selected subset')
        return self.transform(image), self.label_to_incremental[label]


def limit_indices_per_task(indices: Sequence[int], limit: int | None, seed: int) -> list[int]:
    if limit is None:
        return list(indices)
    if limit < 1:
        raise ValueError('limit must be positive')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(list(indices), size=min(limit, len(indices)), replace=False)
    return chosen.tolist()


class CIFAR100OnlineDataset:
    """Convenience wrapper around load_cifar100 for stream and diagnostics management."""

    def __init__(self, root: str = 'artifacts/data', seed: int = 42):
        self.root = root
        self.seed = seed
        self.train_source, self.test_source = load_cifar100(root=root)
        self.class_order = make_class_order(100, seed=seed)
        self.label_mapping = {orig: i for i, orig in enumerate(self.class_order)}
        self.transforms = cifar100_transforms(
            size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation='bicubic',
        )

    def get_train_source(self):
        return self.train_source

    def get_test_source(self):
        return self.test_source

    def get_label_mapping(self) -> dict[int, int]:
        return self.label_mapping

    def get_transforms(self):
        return self.transforms

