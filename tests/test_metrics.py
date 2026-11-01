import math
import pytest
from src.foal.metrics import (
    validate_task_matrix, stage_accuracies, task_forgetting,
    average_forgetting, summary_from_task_matrix,
    format_percentage, format_summary_percentages, aggregate_multi_seed_metrics
)


def test_single_task_no_division_by_zero():
    """T=1: single task must not divide by zero for forgetting."""
    matrix = [[0.82]]
    summary = summary_from_task_matrix(matrix)
    assert summary['average_incremental_accuracy'] == pytest.approx(0.82)
    assert summary['last_accuracy'] == pytest.approx(0.82)
    assert summary['forgetting'] == 0.0
    assert summary['task_forgetting'] == []
    assert summary['num_tasks'] == 1


def test_two_tasks_matches_smoke_test():
    """T=2: matches the 2-task numbers from the smoke test."""
    # Task 1: 0.78
    # Task 2: task 1 drops to 0.69, task 2 is 0.85
    matrix = [
        [0.78],
        [0.69, 0.85]
    ]
    summary = summary_from_task_matrix(matrix)
    assert summary['stage_accuracies'] == pytest.approx([0.78, (0.69 + 0.85) / 2])
    assert summary['average_incremental_accuracy'] == pytest.approx((0.78 + 0.77) / 2)  # 0.775
    assert summary['last_accuracy'] == pytest.approx(0.77)
    assert summary['forgetting'] == pytest.approx(0.78 - 0.69)  # 0.09
    assert summary['task_forgetting'] == pytest.approx([0.09])


def test_three_tasks_hand_calculation():
    """T=3: Standard monotonic test with hand calculations."""
    matrix = [
        [0.90],
        [0.80, 0.70],
        [0.70, 0.65, 0.60]
    ]
    summary = summary_from_task_matrix(matrix)
    
    # Stage 1: 0.90
    # Stage 2: (0.80 + 0.70) / 2 = 0.75
    # Stage 3: (0.70 + 0.65 + 0.60) / 3 = 0.65
    expected_stage_means = [0.90, 0.75, 0.65]
    assert summary['stage_accuracies'] == pytest.approx(expected_stage_means)
    assert summary['average_incremental_accuracy'] == pytest.approx(sum(expected_stage_means) / 3)
    assert summary['last_accuracy'] == pytest.approx(0.65)

    # Forgetting:
    # Task 1: max(0.90, 0.80) - 0.70 = 0.90 - 0.70 = 0.20
    # Task 2: max(0.70) - 0.65 = 0.05
    # Average forgetting: (0.20 + 0.05) / 2 = 0.125
    assert summary['task_forgetting'] == pytest.approx([0.20, 0.05])
    assert summary['forgetting'] == pytest.approx(0.125)


def test_three_tasks_non_monotonic_forgetting():
    """T=3: Non-monotonic performance where task 1 accuracy improves at stage 2 then drops."""
    matrix = [
        [0.70],
        [0.85, 0.80],        # Task 1 improved from 0.70 to 0.85!
        [0.60, 0.75, 0.90]   # Task 1 drops to 0.60, Task 2 drops to 0.75
    ]
    summary = summary_from_task_matrix(matrix)
    
    # Task 1 peak before stage 3 is max(0.70, 0.85) = 0.85
    # Task 1 final is 0.60 -> forgetting = 0.85 - 0.60 = 0.25
    # Task 2 peak is 0.80 -> final 0.75 -> forgetting = 0.05
    # Mean forgetting = (0.25 + 0.05) / 2 = 0.15
    assert summary['task_forgetting'] == pytest.approx([0.25, 0.05])
    assert summary['forgetting'] == pytest.approx(0.15)


def test_positive_backward_transfer():
    """Verify handling when a task ends with higher accuracy than historical peak."""
    matrix = [
        [0.70],
        [0.65, 0.80],
        [0.75, 0.82, 0.85]   # Task 1 ended at 0.75 > max(0.70, 0.65)
    ]
    # Unclipped (signed)
    summary_signed = summary_from_task_matrix(matrix, clip_zero=False)
    # Task 1 forgetting: 0.70 - 0.75 = -0.05
    # Task 2 forgetting: 0.80 - 0.82 = -0.02
    assert summary_signed['task_forgetting'] == pytest.approx([-0.05, -0.02])
    assert summary_signed['forgetting'] == pytest.approx(-0.035)

    # Clipped
    summary_clipped = summary_from_task_matrix(matrix, clip_zero=True)
    assert summary_clipped['task_forgetting'] == pytest.approx([0.0, 0.0])
    assert summary_clipped['forgetting'] == 0.0


def test_validation_errors():
    with pytest.raises(ValueError, match='empty'):
        validate_task_matrix([])

    with pytest.raises(ValueError, match='expected 2 task accuracies, got 1'):
        validate_task_matrix([[0.5], [0.5]])  # Row 2 should have 2 items

    with pytest.raises(ValueError, match='Accuracy must be in range'):
        validate_task_matrix([[1.5]])  # > 1.0

    with pytest.raises(ValueError, match='Accuracy must be in range'):
        validate_task_matrix([[-0.1]])  # < 0.0

    with pytest.raises(ValueError, match='Invalid value'):
        validate_task_matrix([[float('nan')]])


def test_format_percentage():
    assert format_percentage(0.775) == '77.50%'
    assert format_percentage(0.091234, decimals=3) == '9.123%'
    formatted = format_summary_percentages({
        'average_incremental_accuracy': 0.775,
        'last_accuracy': 0.77,
        'forgetting': 0.09
    })
    assert formatted['average_incremental_accuracy'] == '77.50%'
    assert formatted['last_accuracy'] == '77.00%'
    assert formatted['forgetting'] == '9.00%'


def test_aggregate_multi_seed_metrics():
    # 3 seeds
    run1 = {'average_incremental_accuracy': 0.78, 'last_accuracy': 0.75, 'forgetting': 0.08}
    run2 = {'average_incremental_accuracy': 0.80, 'last_accuracy': 0.77, 'forgetting': 0.06}
    run3 = {'average_incremental_accuracy': 0.82, 'last_accuracy': 0.79, 'forgetting': 0.07}

    agg = aggregate_multi_seed_metrics([run1, run2, run3])
    assert agg['num_seeds'] == 3
    # Mean of 0.78, 0.80, 0.82 = 0.80
    assert agg['average_incremental_accuracy_mean'] == pytest.approx(0.80)
    # Sample std of [0.78, 0.80, 0.82] with N-1: sqrt((0.0004 + 0 + 0.0004)/2) = 0.02
    assert agg['average_incremental_accuracy_std'] == pytest.approx(0.02)
    assert agg['last_accuracy_mean'] == pytest.approx(0.77)
    assert agg['last_accuracy_std'] == pytest.approx(0.02)
    assert agg['forgetting_mean'] == pytest.approx(0.07)
    assert agg['forgetting_std'] == pytest.approx(0.01)
    assert agg['average_incremental_accuracy_display'] == '80.00 +/- 2.00% (n=3)'

    # Single seed (n=1): sample std must be None / N/A, never +/- 0.00%
    agg_single = aggregate_multi_seed_metrics([run1])
    assert agg_single['num_seeds'] == 1
    assert agg_single['average_incremental_accuracy_mean'] == pytest.approx(0.78)
    assert agg_single['average_incremental_accuracy_std'] is None
    assert agg_single['average_incremental_accuracy_display'] == '78.00% (n=1, std=N/A)'
    assert '+/- 0.00%' not in agg_single['average_incremental_accuracy_display']
