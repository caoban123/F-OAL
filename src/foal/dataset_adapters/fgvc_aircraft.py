"""FGVC-Aircraft Adapter for Continual Learning.

Benchmark Specifications (F-OAL Table 4 Reproduction Track - Member B):
  - Total classes selected: 100 (out of 102 available aircraft variants)
  - Tasks: 10 incremental stages
  - Classes per task: 10 classes
  - Samples per class: ~33 train samples, ~33 test samples
  - Published Table 4 Targets (A_avg, %):
      * Full F-OAL:           62.2%  (Note: Section 4.4 text states 66.2%, but Table 2 & 4 consistently report 62.2%)
      * w/o Feature Fusion:   60.9%  (-1.3 pp)
      * w/o Smooth Projection: 58.7%  (-3.5 pp)
      * w/o Both:             56.0%  (-6.2 pp)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image

from src.foal.dataset_adapters.base import BaseDatasetAdapter
from src.foal.dataset_adapters.dtd import MockImageSource


class TorchvisionAircraftWrapper:
    """Wraps torchvision.datasets.FGVCAircraft to provide {'img': PIL, 'fine_label': int}."""
    def __init__(self, tv_dataset, selected_classes_map: dict[int, int]):
        self.dataset = tv_dataset
        self.selected_classes_map = selected_classes_map
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


class FGVCAircraftAdapter(BaseDatasetAdapter):
    """Dataset adapter for FGVC-Aircraft continual learning benchmark."""

    # Reference metrics from NeurIPS 2024 Table 4 & Table 2
    REFERENCE_METRICS = {
        "full_foal": 62.2,
        "without_ff": 60.9,
        "without_sp": 58.7,
        "without_both": 56.0,
        "text_discrepancy_section_4_4": 66.2,
    }

    def __init__(
        self,
        *,
        seed: int = 42,
        num_classes: int = 100,
        classes_per_task: int = 10,
        data_root: str | Path = "./data",
        use_mock: bool = False,
    ):
        if num_classes > 102:
            raise ValueError(f"num_classes cannot exceed total available FGVC-Aircraft classes (102)")
        super().__init__(
            dataset_name="fgvc_aircraft",
            num_classes=num_classes,
            classes_per_task=classes_per_task,
            seed=seed,
            data_root=data_root,
        )
        self.use_mock = use_mock

    def load_raw_dataset(self) -> Tuple[Any, Any, Sequence[int], Sequence[int]]:
        """Load FGVC-Aircraft dataset or mock source if offline/mock requested."""
        if self.use_mock:
            return self._create_mock_dataset()

        try:
            from torchvision.datasets import FGVCAircraft
            # 'trainval' is standard training set in FGVCAircraft CL benchmarks
            tv_train = FGVCAircraft(root=str(self.data_root), split="trainval", annotation_level="variant", download=True)
            tv_test = FGVCAircraft(root=str(self.data_root), split="test", annotation_level="variant", download=True)

            raw_to_mapped = {i: i for i in range(self.num_classes)}
            train_src = TorchvisionAircraftWrapper(tv_train, raw_to_mapped)
            test_src = TorchvisionAircraftWrapper(tv_test, raw_to_mapped)
            return train_src, test_src, train_src.fine_label, test_src.fine_label
        except Exception as exc:
            print(f"Notice: FGVC-Aircraft not found or download failed ({exc}). Falling back to MockImageSource.")
            return self._create_mock_dataset()

    def _create_mock_dataset(self) -> Tuple[Any, Any, Sequence[int], Sequence[int]]:
        """Create deterministic mock dataset: ~33 train & ~33 test samples per class."""
        samples_per_class = 33
        train_labels = []
        test_labels = []
        for c in range(self.num_classes):
            train_labels.extend([c] * samples_per_class)
            test_labels.extend([c] * samples_per_class)

        train_src = MockImageSource(len(train_labels), train_labels)
        test_src = MockImageSource(len(test_labels), test_labels)
        return train_src, test_src, train_labels, test_labels
