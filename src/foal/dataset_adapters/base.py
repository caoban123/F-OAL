"""Shared Dataset Adapter Protocol for Continual Learning benchmarks in F-OAL.

This module provides an extensible, contract-enforcing foundation for benchmark
datasets (CIFAR-100, DTD, FGVC-Aircraft, etc.) ensuring:
  1. Strict class-incremental ordering without data leakage.
  2. Single-pass online streaming contracts.
  3. Consistent TaskSpec schemas compatible with F-OAL model runners and baselines.
  4. Cryptographic SHA-256 fingerprinting of train/test task splits.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from src.foal.datasets import (
    TaskSpec,
    IndexedImageDataset,
    make_class_order,
    make_task_specs,
    limit_indices_per_task,
    cifar100_transforms,
)


@dataclass
class DatasetSplitMetadata:
    """Metadata describing dataset splits and validation checksums."""
    dataset_name: str
    num_classes: int
    classes_per_task: int
    num_tasks: int
    num_train_samples: int
    num_test_samples: int
    class_order: list[int]
    split_fingerprint_sha256: str
    custom_attributes: dict[str, Any] = field(default_factory=dict)


class BaseDatasetAdapter(ABC):
    """Abstract Base Class for Continual Learning Dataset Adapters."""

    def __init__(
        self,
        *,
        dataset_name: str,
        num_classes: int,
        classes_per_task: int,
        seed: int = 42,
        explicit_order: Sequence[int] | None = None,
        data_root: str | Path = "./data",
    ):
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if classes_per_task < 1 or num_classes % classes_per_task != 0:
            raise ValueError(
                f"classes_per_task ({classes_per_task}) must evenly divide num_classes ({num_classes})"
            )

        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.classes_per_task = classes_per_task
        self.num_tasks = num_classes // classes_per_task
        self.seed = seed
        self.data_root = Path(data_root)

        # Generate or assign class order permutation
        self.order: list[int] = make_class_order(
            num_classes=self.num_classes,
            seed=self.seed,
            explicit=explicit_order,
        )

        self._train_source: Any = None
        self._test_source: Any = None
        self._train_labels: list[int] = []
        self._test_labels: list[int] = []
        self._task_specs: list[TaskSpec] = []
        self._initialized: bool = False

    @abstractmethod
    def load_raw_dataset(self) -> Tuple[Any, Any, Sequence[int], Sequence[int]]:
        """Load or mock raw dataset sources and integer labels.

        Returns:
            (train_source, test_source, train_labels, test_labels)
        """
        pass

    def setup(self) -> None:
        """Initialize dataset sources, label arrays, and task specifications."""
        if self._initialized:
            return

        train_src, test_src, train_lbls, test_lbls = self.load_raw_dataset()
        self._train_source = train_src
        self._test_source = test_src
        self._train_labels = list(train_lbls)
        self._test_labels = list(test_lbls)

        # Enforce strict task specs using core F-OAL protocol
        self._task_specs = make_task_specs(
            train_labels=self._train_labels,
            test_labels=self._test_labels,
            order=self.order,
            classes_per_task=self.classes_per_task,
        )
        self._initialized = True

    @property
    def task_specs(self) -> list[TaskSpec]:
        """Return the precomputed TaskSpec list for all incremental stages."""
        if not self._initialized:
            self.setup()
        return self._task_specs

    @property
    def train_labels(self) -> list[int]:
        if not self._initialized:
            self.setup()
        return self._train_labels

    @property
    def test_labels(self) -> list[int]:
        if not self._initialized:
            self.setup()
        return self._test_labels

    def get_transforms(
        self,
        size: int = 224,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        interpolation: str = "bicubic",
        normalize: bool = True,
    ) -> Callable:
        """Return standard preprocessing transforms matching ViT backbone requirements."""
        return cifar100_transforms(
            size=size,
            mean=mean,
            std=std,
            interpolation=interpolation,
            normalize=normalize,
        )

    def create_task_datasets(
        self,
        task_id: int,
        transform: Callable,
        limit_train: int | None = None,
        limit_test: int | None = None,
        limit_seed: int = 42,
    ) -> Tuple[IndexedImageDataset, IndexedImageDataset]:
        """Create PyTorch IndexedImageDataset objects for training and cumulative test."""
        if not self._initialized:
            self.setup()

        spec = self._task_specs[task_id]

        # Restrict training indices to active task
        train_indices = list(spec.train_indices)
        if limit_train is not None:
            train_indices = limit_indices_per_task(
                train_indices, limit_train, limit_seed + task_id
            )

        # Restrict cumulative seen test indices
        test_indices = list(spec.seen_test_indices)
        if limit_test is not None:
            test_indices = limit_indices_per_task(
                test_indices, limit_test, limit_seed + task_id
            )

        # Build label mapping from original dataset class id to incremental [0, total_seen-1]
        seen_order = self.order[: spec.total_seen_classes]
        label_to_incremental = {orig_label: inc_idx for inc_idx, orig_label in enumerate(seen_order)}

        train_ds = IndexedImageDataset(
            source=self._train_source,
            indices=train_indices,
            label_to_incremental=label_to_incremental,
            transform=transform,
        )
        test_ds = IndexedImageDataset(
            source=self._test_source,
            indices=test_indices,
            label_to_incremental=label_to_incremental,
            transform=transform,
        )
        return train_ds, test_ds

    def compute_fingerprint(
        self,
        max_tasks: int | None = None,
        limit_train: int | None = None,
        limit_test: int | None = None,
    ) -> dict[str, Any]:
        """Compute cryptographic SHA-256 fingerprint of the split configuration."""
        if not self._initialized:
            self.setup()

        active_tasks = max_tasks if max_tasks is not None else self.num_tasks
        task_entries = []
        for spec in self._task_specs[:active_tasks]:
            train_ids = list(spec.train_indices)
            if limit_train is not None:
                train_ids = limit_indices_per_task(train_ids, limit_train, self.seed + spec.task_id)

            test_ids = list(spec.seen_test_indices)
            if limit_test is not None:
                test_ids = limit_indices_per_task(test_ids, limit_test, self.seed + spec.task_id)

            train_sha = hashlib.sha256(json.dumps(train_ids).encode("utf-8")).hexdigest()
            test_sha = hashlib.sha256(json.dumps(test_ids).encode("utf-8")).hexdigest()

            task_entries.append({
                "task_id": int(spec.task_id),
                "new_original_labels": list(spec.new_original_labels),
                "num_train_samples": len(train_ids),
                "num_test_samples": len(test_ids),
                "train_indices_sha256": train_sha,
                "test_indices_sha256": test_sha,
            })

        payload = {
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "classes_per_task": self.classes_per_task,
            "num_tasks": active_tasks,
            "class_order": list(self.order),
            "seed": int(self.seed),
            "limit_train": limit_train,
            "limit_test": limit_test,
            "tasks": task_entries,
        }
        overall_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "dataset_name": self.dataset_name,
            "sha256": overall_sha,
            "class_order_sha256": hashlib.sha256(json.dumps(self.order).encode("utf-8")).hexdigest(),
            "num_tasks": active_tasks,
            "tasks": task_entries,
        }

    def get_metadata(self) -> DatasetSplitMetadata:
        """Return standardized metadata object."""
        if not self._initialized:
            self.setup()
        fp = self.compute_fingerprint()
        return DatasetSplitMetadata(
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            classes_per_task=self.classes_per_task,
            num_tasks=self.num_tasks,
            num_train_samples=len(self._train_labels),
            num_test_samples=len(self._test_labels),
            class_order=self.order,
            split_fingerprint_sha256=fp["sha256"],
        )
