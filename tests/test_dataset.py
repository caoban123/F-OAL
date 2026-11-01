from src.foal.datasets import make_class_order, make_task_specs


def test_seeded_order_is_reproducible_and_permutation():
    a = make_class_order(10, seed=42)
    assert a == make_class_order(10, seed=42)
    assert sorted(a) == list(range(10))


def test_task_partition_no_train_replay_and_no_future_test():
    train_labels = [0, 1, 2, 3, 0, 1, 2, 3]
    test_labels = [0, 1, 2, 3]
    specs = make_task_specs(train_labels, test_labels, order=[2, 0, 3, 1], classes_per_task=2)
    assert len(specs) == 2
    assert specs[0].new_original_labels == (2, 0)
    assert set(specs[0].train_indices).isdisjoint(set(specs[1].train_indices))
    assert set(specs[0].seen_test_indices) == {0, 2}
    assert set(specs[1].seen_test_indices) == {0, 1, 2, 3}
    assert specs[0].total_seen_classes == 2
    assert specs[1].total_seen_classes == 4
