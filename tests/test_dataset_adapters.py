"""Unit tests for Continual Learning Dataset Adapters (DTD, FGVC-Aircraft, Base protocol)."""
from pathlib import Path
import pytest
import torch

from src.foal.dataset_adapters.base import BaseDatasetAdapter
from src.foal.dataset_adapters.dtd import DTDAdapter, DTD_ALL_CLASSES
from src.foal.dataset_adapters.fgvc_aircraft import FGVCAircraftAdapter


def test_dtd_adapter_specs_and_invariants():
    """Verify DTDAdapter correctly partitions 40 classes into 10 tasks without leakage."""
    adapter = DTDAdapter(num_classes=40, classes_per_task=4, seed=42, use_mock=True)
    adapter.setup()

    assert adapter.num_tasks == 10
    assert len(adapter.task_specs) == 10
    assert len(adapter.order) == 40
    assert len(set(adapter.order)) == 40

    # Invariant 1: No training sample replay across any tasks
    all_train_indices = set()
    seen_classes = set()
    for spec in adapter.task_specs:
        train_set = set(spec.train_indices)
        assert len(train_set.intersection(all_train_indices)) == 0, "Data leakage: train index replayed!"
        all_train_indices.update(train_set)

        # Invariant 2: Total seen classes progression
        new_classes = set(spec.new_original_labels)
        assert len(new_classes) == 4
        seen_classes.update(new_classes)
        assert spec.total_seen_classes == len(seen_classes)

        # Invariant 3: Test indices only contain seen classes
        for test_idx in spec.seen_test_indices:
            lbl = adapter.test_labels[test_idx]
            assert lbl in seen_classes, f"Data leakage: test label {lbl} exposed before introduced!"


def test_dtd_adapter_reference_metrics():
    """Verify DTD published reference metrics match NeurIPS 2024 Table 4."""
    assert DTDAdapter.REFERENCE_METRICS["full_foal"] == 82.8
    assert DTDAdapter.REFERENCE_METRICS["without_ff"] == 80.5
    assert DTDAdapter.REFERENCE_METRICS["without_sp"] == 79.3
    assert DTDAdapter.REFERENCE_METRICS["without_both"] == 71.2


def test_fgvc_aircraft_adapter_specs_and_invariants():
    """Verify FGVCAircraftAdapter partitions 100 classes into 10 tasks without leakage."""
    adapter = FGVCAircraftAdapter(num_classes=100, classes_per_task=10, seed=42, use_mock=True)
    adapter.setup()

    assert adapter.num_tasks == 10
    assert len(adapter.task_specs) == 10
    assert len(adapter.order) == 100

    all_train_indices = set()
    seen_classes = set()
    for spec in adapter.task_specs:
        train_set = set(spec.train_indices)
        assert len(train_set.intersection(all_train_indices)) == 0, "Data leakage: train index replayed!"
        all_train_indices.update(train_set)

        new_classes = set(spec.new_original_labels)
        assert len(new_classes) == 10
        seen_classes.update(new_classes)
        assert spec.total_seen_classes == len(seen_classes)

        for test_idx in spec.seen_test_indices:
            lbl = adapter.test_labels[test_idx]
            assert lbl in seen_classes


def test_fgvc_aircraft_reference_metrics():
    """Verify Aircraft published reference metrics match Table 4 & document Section 4.4 discrepancy."""
    assert FGVCAircraftAdapter.REFERENCE_METRICS["full_foal"] == 62.2
    assert FGVCAircraftAdapter.REFERENCE_METRICS["without_ff"] == 60.9
    assert FGVCAircraftAdapter.REFERENCE_METRICS["without_sp"] == 58.7
    assert FGVCAircraftAdapter.REFERENCE_METRICS["without_both"] == 56.0
    assert FGVCAircraftAdapter.REFERENCE_METRICS["text_discrepancy_section_4_4"] == 66.2


def test_adapter_deterministic_fingerprint():
    """Verify that fingerprinting is deterministic across multiple calls with same seed."""
    adapter1 = DTDAdapter(num_classes=40, classes_per_task=4, seed=42, use_mock=True)
    adapter2 = DTDAdapter(num_classes=40, classes_per_task=4, seed=42, use_mock=True)

    fp1 = adapter1.compute_fingerprint()
    fp2 = adapter2.compute_fingerprint()

    assert fp1["sha256"] == fp2["sha256"]
    assert fp1["class_order_sha256"] == fp2["class_order_sha256"]
