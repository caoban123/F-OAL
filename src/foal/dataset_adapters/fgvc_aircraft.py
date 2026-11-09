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


HF_TRAIN_REPO = "Multimodal-Fatima/FGVC_Aircraft_train"
HF_TEST_REPO = "Multimodal-Fatima/FGVC_Aircraft_test"
EXPECTED_TRAIN, EXPECTED_TEST, EXPECTED_CLASSES = 3334, 3333, 100


def load_fgvc_aircraft() -> Tuple[Any, Any]:
    """Load the official FGVC-Aircraft `train` and `test` splits from Hugging Face.

    The paper reports about 33 train and 33 test images per class, which is the
    `train` split (3,334 images), not `trainval` (6,667 images).

    Returns:
        (train, test) datasets exposing the columns `img` and `fine_label`.

    Raises:
        ValueError: if the downloaded data does not match the expected sizes.
    """
    from datasets import load_dataset

    sources = []
    for repo, split, expected in ((HF_TRAIN_REPO, "train", EXPECTED_TRAIN), (HF_TEST_REPO, "test", EXPECTED_TEST)):
        data = load_dataset(repo, split=split).select_columns(["image", "label"])
        data = data.rename_columns({"image": "img", "label": "fine_label"})
        n_classes = len(set(data["fine_label"]))
        if len(data) != expected or n_classes != EXPECTED_CLASSES:
            raise ValueError(f"{repo}: got {len(data)} images and {n_classes} classes")
        sources.append(data)
    return sources[0], sources[1]


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

        # No fallback to mock data: a failed download must stop the run.
        train_src, test_src = load_fgvc_aircraft()
        return train_src, test_src, list(train_src["fine_label"]), list(test_src["fine_label"])

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
