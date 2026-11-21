"""DTD (Describable Textures Dataset) Adapter for Continual Learning.

Benchmark Specifications (F-OAL Table 4 Reproduction Track - Member A):
  - Total classes selected: 40 (out of 47 available in DTD)
  - Tasks: 10 incremental stages
  - Classes per task: 4 classes
  - Samples per class: 40 train samples, 40 test samples
  - Published Table 4 Targets (A_avg, %):
      * Full F-OAL:           82.8%
      * w/o Feature Fusion:   80.5%  (-2.3 pp)
      * w/o Smooth Projection: 79.3%  (-3.5 pp)
      * w/o Both:             71.2%  (-11.6 pp)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image

from src.foal.dataset_adapters.base import BaseDatasetAdapter
from src.foal.dtd_datasets import DTD_CLASS_NAMES, select_dtd_classes


# Official 47 texture classes aligned identically with src/foal/dtd_datasets.py (tanganke/dtd & torchvision DTD)
DTD_ALL_CLASSES = list(DTD_CLASS_NAMES)


class MockImageSource:
    """Mock image source providing PIL Images for unit tests and offline environments."""
    def __init__(self, num_samples: int, labels: Sequence[int], image_size: Tuple[int, int] = (64, 64)):
        self.num_samples = num_samples
        self.fine_label = list(labels)
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Return a simple deterministic solid-color or patterned PIL image
        val = int(self.fine_label[idx] * 5 % 255)
        img = Image.new("RGB", self.image_size, color=(val, (val * 2) % 255, (val * 3) % 255))
        return {"img": img, "fine_label": int(self.fine_label[idx])}


class TorchvisionDTDWrapper:
    """Wraps torchvision.datasets.DTD to provide a dictionary interface {'img': PIL, 'fine_label': int}."""
    def __init__(self, tv_dataset, selected_classes_map: dict[int, int]):
        self.dataset = tv_dataset
        self.selected_classes_map = selected_classes_map
        # Filter indices to only include samples from the selected classes
        self.valid_indices = []
        self.fine_label = []
        for i in range(len(tv_dataset)):
            raw_label = int(tv_dataset._labels[i]) if hasattr(tv_dataset, "_labels") else int(tv_dataset[i][1])
            if raw_label in selected_classes_map:
                mapped_label = selected_classes_map[raw_label]
                self.valid_indices.append(i)
                self.fine_label.append(mapped_label)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self.valid_indices[idx]
        img, _ = self.dataset[real_idx]
        return {"img": img, "fine_label": self.fine_label[idx]}


class DTDAdapter(BaseDatasetAdapter):
    """Dataset adapter for DTD continual learning benchmark."""

    # Reference metrics from NeurIPS 2024 Table 4
    REFERENCE_METRICS = {
        "full_foal": 82.8,
        "without_ff": 80.5,
        "without_sp": 79.3,
        "without_both": 71.2,
    }

    def __init__(
        self,
        *,
        seed: int = 42,
        num_classes: int = 40,
        classes_per_task: int = 4,
        samples_per_class: int | None = 40,
        data_root: str | Path = "./data",
        use_mock: bool = False,
    ):
        if num_classes > len(DTD_ALL_CLASSES):
            raise ValueError(f"num_classes cannot exceed total available DTD classes ({len(DTD_ALL_CLASSES)})")
        super().__init__(
            dataset_name="dtd",
            num_classes=num_classes,
            classes_per_task=classes_per_task,
            seed=seed,
            data_root=data_root,
        )
        self.use_mock = use_mock
        self.samples_per_class = samples_per_class
        # Select classes using the exact same seeded selector as src/foal/dtd_datasets.py
        self.selected_class_indices = select_dtd_classes(
            num_classes_to_use=self.num_classes,
            total_classes=len(DTD_ALL_CLASSES),
            seed=self.seed,
        )
        self.selected_classes = [DTD_ALL_CLASSES[i] for i in self.selected_class_indices]

    def load_raw_dataset(self) -> Tuple[Any, Any, Sequence[int], Sequence[int]]:
        """Load DTD dataset or generate mock source if offline/mock requested."""
        if self.use_mock:
            return self._create_mock_dataset()

        try:
            from torchvision.datasets import DTD
            tv_train = DTD(root=str(self.data_root), split="train", download=True)
            tv_test = DTD(root=str(self.data_root), split="test", download=True)

            # Map selected raw DTD class indices to [0, num_classes-1]
            raw_to_mapped = {
                orig_cls: mapped_idx
                for mapped_idx, orig_cls in enumerate(self.selected_class_indices)
            }
            train_src = TorchvisionDTDWrapper(tv_train, raw_to_mapped)
            test_src = TorchvisionDTDWrapper(tv_test, raw_to_mapped)
            return train_src, test_src, train_src.fine_label, test_src.fine_label
        except Exception as exc:
            # Fallback to mock for CI/offline environments if dataset not downloaded
            print(f"Notice: DTD not found or download failed ({exc}). Falling back to MockImageSource for DTD.")
            return self._create_mock_dataset()

    def _create_mock_dataset(self) -> Tuple[Any, Any, Sequence[int], Sequence[int]]:
        """Create deterministic mock dataset: samples_per_class (default 40) train & 40 test samples."""
        train_spc = self.samples_per_class if self.samples_per_class is not None else 40
        test_spc = 40
        train_labels = []
        test_labels = []
        for c in range(self.num_classes):
            train_labels.extend([c] * train_spc)
            test_labels.extend([c] * test_spc)

        train_src = MockImageSource(len(train_labels), train_labels)
        test_src = MockImageSource(len(test_labels), test_labels)
        return train_src, test_src, train_labels, test_labels
