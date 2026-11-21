import pytest
from src.foal.dtd_datasets import (
    select_dtd_classes,
    make_dtd_class_order,
    make_dtd_task_specs,
    filter_indices_by_classes,
    DTD_CLASS_NAMES,
)


def test_select_dtd_classes():
    selected = select_dtd_classes(num_classes_to_use=40, total_classes=47, seed=42)
    assert len(selected) == 40
    assert len(set(selected)) == 40
    assert all(0 <= c < 47 for c in selected)
    # Reproducible
    assert selected == select_dtd_classes(num_classes_to_use=40, total_classes=47, seed=42)


def test_make_dtd_class_order():
    order = make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=42)
    assert len(order) == 40
    assert len(set(order)) == 40
    # Reproducible
    assert order == make_dtd_class_order(num_classes_to_use=40, total_classes=47, seed=42)


def test_dtd_task_partition_zero_replay():
    order = list(range(40))
    # Synthetic labels
    train_labels = [c for c in range(40) for _ in range(10)]
    test_labels = [c for c in range(40) for _ in range(5)]

    specs = make_dtd_task_specs(train_labels, test_labels, order=order, classes_per_task=4)
    assert len(specs) == 10

    # Ensure zero replay: all train indices partitioned disjointly
    all_train = []
    for spec in specs:
        assert len(spec.new_original_labels) == 4
        all_train.extend(spec.train_indices)

    assert len(all_train) == len(set(all_train))
    assert len(all_train) == 400

    # Test indices cumulative
    for i, spec in enumerate(specs):
        expected_classes = (i + 1) * 4
        assert spec.total_seen_classes == expected_classes
        assert len(spec.seen_test_indices) == expected_classes * 5


def test_filter_indices_by_classes():
    labels = [10, 20, 30, 40, 50, 10, 20]
    indices = filter_indices_by_classes(labels, {10, 30})
    assert indices == [0, 2, 5]


def test_dtd_class_names():
    assert len(DTD_CLASS_NAMES) == 47
    assert 'banded' in DTD_CLASS_NAMES
    assert 'zigzagged' in DTD_CLASS_NAMES
