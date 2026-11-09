"""DTD (Describable Textures Dataset) task stream for F-OAL experiments.

DTD has 47 texture categories with 120 images each (5,640 total).
The tanganke/dtd dataset on HF provides: 3,760 train / 1,880 test.
(train = 80 images/class, test = 40 images/class)

Following the F-OAL paper protocol:
  - Use 40 out of 47 classes (to make 10 evenly-divided tasks)
  - 10 tasks × 4 classes per task = 40 classes
  - Train: 40 images per class per task (subsample from ~80)
  - Test: 40 images per class
  - Batch size: 10
  - One-pass streaming, no replay

Paper reference results (Table 2 & 4):
  - Full F-OAL:        A_avg = 82.8%
  - w/o Feature Fusion: A_avg = 80.5%
  - w/o Smooth Proj:    A_avg = 79.3%
  - w/o Both:           A_avg = 71.2%
  - Training time:      ~8 seconds

Label names (0-46): banded, blotchy, braided, bubbly, bumpy, chequered,
cobwebbed, cracked, crosshatched, crystalline, dotted, fibrous, flecked,
freckled, frilly, gauzy, grid, grooved, honeycombed, interlaced, knitted,
lacelike, lined, marbled, matted, meshed, paisley, perforated, pitted,
pleated, polka-dotted, porous, potholed, scaly, smeared, spiralled,
sprinkled, stained, stratified, striped, studded, swirly, veined, waffled,
woven, wrinkled, zigzagged.
"""
from __future__ import annotations
import os
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Tuple, Any, Dict, Optional
import numpy as np
import torch
from PIL import Image

from .datasets import (
    make_class_order,
    make_task_specs,
    cifar100_transforms,
    limit_indices_per_task,
    TaskSpec,
)


# DTD class names for reference and reporting
DTD_CLASS_NAMES = [
    'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered',
    'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted',
    'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid',
    'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike',
    'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated',
    'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly',
    'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
    'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven',
    'wrinkled', 'zigzagged',
]


class HFDTDAdapter:
    """Adapter to make HuggingFace DTD dataset compatible with CIFAR-100 interface."""
    def __init__(self, hf_dataset):
        self._dataset = hf_dataset
        self.fine_label = list(hf_dataset['label'])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx in ('fine_label', 'label'):
                return self.fine_label
            raise KeyError(idx)
        sample = self._dataset[int(idx)]
        img = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return {
            'img': img,
            'fine_label': sample['label'],
        }


class TorchvisionDTDAdapter:
    """Adapter to make Torchvision DTD dataset compatible with CIFAR-100 interface."""
    def __init__(self, splits, root_dir):
        from torchvision.datasets import DTD as TorchvisionDTD
        self._images = []
        self._labels = []
        for split in splits:
            ds = TorchvisionDTD(root=root_dir, split=split, download=True)
            self._images.extend(list(ds._image_files))
            self._labels.extend(list(ds._labels))
        self.fine_label = self._labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx in ('fine_label', 'label'):
                return self.fine_label
            raise KeyError(idx)
        img = Image.open(self._images[idx]).convert('RGB')
        return {'img': img, 'fine_label': self._labels[idx]}


def load_dtd(root: str = './data'):
    """Download and load DTD dataset from Hugging Face (tanganke/dtd).

    Returns:
        (train_source, test_source) where each supports indexing and has
        'fine_label' field for consistency with CIFAR-100 code.
    """
    try:
        from datasets import load_dataset
        try:
            data = load_dataset('tanganke/dtd')
        except Exception:
            if os.environ.get('HF_DATASETS_OFFLINE') == '1':
                os.environ['HF_DATASETS_OFFLINE'] = '0'
                data = load_dataset('tanganke/dtd')
            else:
                raise
        train_data = data['train']   # 3,760 samples
        test_data = data['test']     # 1,880 samples

        return HFDTDAdapter(train_data), HFDTDAdapter(test_data)

    except Exception as exc:
        # Fallback to torchvision DTD
        try:
            train_adapter = TorchvisionDTDAdapter(['train', 'val'], root)
            test_adapter = TorchvisionDTDAdapter(['test'], root)
            return train_adapter, test_adapter
        except Exception as exc2:
            raise RuntimeError(
                f"Failed to load DTD from both HF and torchvision. "
                f"HF error: {exc}, Torchvision error: {exc2}"
            ) from exc2


def select_dtd_classes(num_classes_to_use: int = 40, total_classes: int = 47,
                       seed: int = 42) -> list[int]:
    """Select a subset of DTD classes for the experiment.

    Following the F-OAL paper: select 40 out of 47 classes to make
    10 evenly-divided tasks of 4 classes each.

    Returns:
        List of selected class indices (sorted for reproducibility).
    """
    rng = np.random.default_rng(seed)
    all_classes = list(range(total_classes))
    selected = rng.choice(all_classes, size=num_classes_to_use, replace=False)
    return sorted(selected.tolist())


def make_dtd_task_specs(train_labels: Sequence[int], test_labels: Sequence[int],
                        order: Sequence[int], classes_per_task: int = 4) -> list[TaskSpec]:
    """Create task specifications for DTD.

    With 40 selected classes and classes_per_task=4:
    - 10 tasks × 4 classes each = 40 classes (divides evenly)
    """
    num_classes = len(order)
    if classes_per_task < 1:
        raise ValueError('classes_per_task must be positive')
    if num_classes % classes_per_task != 0:
        raise ValueError(
            f'num_classes ({num_classes}) must be divisible by '
            f'classes_per_task ({classes_per_task})'
        )

    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    if train.ndim != 1 or test.ndim != 1:
        raise ValueError('Labels must be 1D')

    specs = []
    seen = set()
    all_train_indices = set()

    for task_id, start in enumerate(range(0, num_classes, classes_per_task)):
        end = start + classes_per_task
        new_labels = tuple(order[start:end])
        new_set = set(new_labels)

        train_ids = np.flatnonzero(np.isin(train, list(new_set))).tolist()
        if any(i in all_train_indices for i in train_ids):
            raise AssertionError('A training sample would be replayed')
        all_train_indices.update(train_ids)

        seen.update(new_set)
        test_ids = np.flatnonzero(np.isin(test, list(seen))).tolist()

        specs.append(TaskSpec(
            task_id=task_id,
            new_original_labels=new_labels,
            train_indices=tuple(train_ids),
            seen_test_indices=tuple(test_ids),
            total_seen_classes=len(seen),
        ))

    return specs


def make_dtd_class_order(num_classes_to_use: int = 40, total_classes: int = 47,
                         seed: int = 42) -> list[int]:
    """Generate a class ordering for DTD experiments.

    First selects which classes to use, then permutes them.
    Uses separate RNG operations for selection and permutation
    to ensure reproducibility.

    Args:
        num_classes_to_use: How many classes to include (default: 40 per paper).
        total_classes: Total DTD classes (47).
        seed: Random seed.

    Returns:
        Permuted list of selected class indices.
    """
    selected = select_dtd_classes(num_classes_to_use, total_classes, seed)
    # Permute the selected classes using a separate RNG stream
    rng = np.random.default_rng(seed + 1000)  # offset to avoid correlation
    permuted = rng.permutation(selected).tolist()
    return permuted


def filter_indices_by_classes(labels: Sequence[int],
                              selected_classes: set[int]) -> list[int]:
    """Return indices of samples whose labels are in selected_classes."""
    labels_arr = np.asarray(labels)
    return np.flatnonzero(np.isin(labels_arr, list(selected_classes))).tolist()
