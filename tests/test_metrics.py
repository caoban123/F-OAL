import json
import math
from pathlib import Path
import pytest
from src.foal.metrics import (
    validate_task_matrix,
    stage_accuracies,
    task_forgetting,
    average_forgetting,
    task_forgetting_past_tasks,
    forgetting_past_tasks,
    task_forgetting_official_code,
    forgetting_official_code,
    summary_from_task_matrix,
    format_percentage,
    format_summary_percentages,
    aggregate_multi_seed_metrics,
)


def test_single_task_no_division_by_zero():
    """T=1: single task must not divide by zero for forgetting."""
    matrix = [[0.82]]
    summary = summary_from_task_matrix(matrix)
    assert summary['average_incremental_accuracy'] == pytest.approx(0.82)
    assert summary['last_accuracy'] == pytest.approx(0.82)
    assert summary['forgetting'] == 0.0
    assert summary['forgetting_past_tasks'] == 0.0
    assert summary['forgetting_official_code'] == 0.0
    assert summary['task_forgetting'] == []
    assert summary['task_forgetting_past_tasks'] == []
    assert summary['task_forgetting_official_code'] == [0.0]
    assert summary['num_tasks'] == 1


def test_two_tasks_matches_smoke_test():
    """Case 4 (T=2) & Case 1 (Decreasing accuracy): matches the 2-task numbers from the smoke test."""
    # Task 1: 0.78 -> drops to 0.69 at stage 2; Task 2: 0.85
    matrix = [
        [0.78],
        [0.69, 0.85],
    ]
    summary = summary_from_task_matrix(matrix)
    assert summary['stage_accuracies'] == pytest.approx([0.78, (0.69 + 0.85) / 2])
    assert summary['average_incremental_accuracy'] == pytest.approx((0.78 + 0.77) / 2)  # 0.775
    assert summary['last_accuracy'] == pytest.approx(0.77)
    # Past-task forgetting: (0.78 - 0.69) / 1 = 0.09
    assert summary['forgetting'] == pytest.approx(0.09)
    assert summary['forgetting_past_tasks'] == pytest.approx(0.09)
    assert summary['task_forgetting'] == pytest.approx([0.09])
    assert summary['task_forgetting_past_tasks'] == pytest.approx([0.09])
    # Official-code forgetting: ((0.78 - 0.69) + (0.85 - 0.85)) / 2 = 0.045
    assert summary['forgetting_official_code'] == pytest.approx(0.045)
    assert summary['task_forgetting_official_code'] == pytest.approx([0.09, 0.0])


def test_task_with_unchanged_accuracy():
    """Case 3: A task whose accuracy remains unchanged across all stages has zero forgetting under both formulas."""
    matrix = [
        [0.80],
        [0.80, 0.75],
        [0.80, 0.75, 0.90],
    ]
    summary = summary_from_task_matrix(matrix)
    assert summary['task_forgetting_past_tasks'] == pytest.approx([0.0, 0.0])
    assert summary['forgetting_past_tasks'] == pytest.approx(0.0)
    assert summary['task_forgetting_official_code'] == pytest.approx([0.0, 0.0, 0.0])
    assert summary['forgetting_official_code'] == pytest.approx(0.0)


def test_three_tasks_hand_calculation():
    """T=3: Standard monotonic decreasing task accuracy with hand calculations."""
    matrix = [
        [0.90],
        [0.80, 0.70],
        [0.70, 0.65, 0.60],
    ]
    summary = summary_from_task_matrix(matrix)

    expected_stage_means = [0.90, 0.75, 0.65]
    assert summary['stage_accuracies'] == pytest.approx(expected_stage_means)
    assert summary['average_incremental_accuracy'] == pytest.approx(sum(expected_stage_means) / 3)
    assert summary['last_accuracy'] == pytest.approx(0.65)

    # Past-task forgetting: Task 1 = 0.20, Task 2 = 0.05 -> mean = 0.125
    assert summary['task_forgetting'] == pytest.approx([0.20, 0.05])
    assert summary['forgetting_past_tasks'] == pytest.approx(0.125)
    # Official-code forgetting: Task 1 = 0.20, Task 2 = 0.05, Task 3 = 0.0 -> mean = 0.25 / 3
    assert summary['task_forgetting_official_code'] == pytest.approx([0.20, 0.05, 0.0])
    assert summary['forgetting_official_code'] == pytest.approx(0.25 / 3)


def test_three_tasks_non_monotonic_forgetting():
    """T=3: Non-monotonic performance where task 1 accuracy improves at stage 2 then drops."""
    matrix = [
        [0.70],
        [0.85, 0.80],        # Task 1 improved from 0.70 to 0.85!
        [0.60, 0.75, 0.90],  # Task 1 drops to 0.60, Task 2 drops to 0.75
    ]
    summary = summary_from_task_matrix(matrix)

    assert summary['task_forgetting_past_tasks'] == pytest.approx([0.25, 0.05])
    assert summary['forgetting_past_tasks'] == pytest.approx(0.15)
    assert summary['task_forgetting_official_code'] == pytest.approx([0.25, 0.05, 0.0])
    assert summary['forgetting_official_code'] == pytest.approx(0.10)


def test_positive_backward_transfer():
    """Case 2: Verify handling when a task ends with higher accuracy than historical peak."""
    matrix = [
        [0.70],
        [0.65, 0.80],
        [0.75, 0.82, 0.85],  # Task 1 ended at 0.75 > max(0.70, 0.65); Task 2 ended at 0.82 > 0.80
    ]
    # Unclipped (signed) past-task forgetting
    summary_signed = summary_from_task_matrix(matrix, clip_zero=False)
    assert summary_signed['task_forgetting'] == pytest.approx([-0.05, -0.02])
    assert summary_signed['task_forgetting_past_tasks'] == pytest.approx([-0.05, -0.02])
    assert summary_signed['forgetting'] == pytest.approx(-0.035)
    assert summary_signed['forgetting_past_tasks'] == pytest.approx(-0.035)

    # Official-code forgetting includes stage T in max, clamping negative forgetting to 0.0
    assert summary_signed['task_forgetting_official_code'] == pytest.approx([0.0, 0.0, 0.0])
    assert summary_signed['forgetting_official_code'] == pytest.approx(0.0)

    # Clipped past-task forgetting
    summary_clipped = summary_from_task_matrix(matrix, clip_zero=True)
    assert summary_clipped['task_forgetting_past_tasks'] == pytest.approx([0.0, 0.0])
    assert summary_clipped['forgetting_past_tasks'] == 0.0


def test_ten_tasks_synthetic_matrix():
    """Case 5 (T=10): Verify both forgetting formulas on a 10-task matrix."""
    # Construct a 10-task matrix where task j starts at 0.90 and drops by 0.01 at each subsequent stage
    matrix = []
    for k in range(10):
        row = [0.90 - 0.01 * (k - j) for j in range(k + 1)]
        matrix.append(row)
    summary = summary_from_task_matrix(matrix)
    assert summary['num_tasks'] == 10
    # For each past task j in 0..8, peak is 0.90, final is 0.90 - 0.01 * (9 - j) -> drop is 0.01 * (9 - j)
    expected_past_drops = [0.01 * (9 - j) for j in range(9)]  # [0.09, 0.08, ..., 0.01], sum = 0.45
    assert summary['task_forgetting_past_tasks'] == pytest.approx(expected_past_drops)
    assert summary['forgetting_past_tasks'] == pytest.approx(0.45 / 9)  # 0.05
    assert summary['task_forgetting_official_code'] == pytest.approx(expected_past_drops + [0.0])
    assert summary['forgetting_official_code'] == pytest.approx(0.45 / 10)  # 0.045


def test_historical_official_and_our_matrices_c0_verification():
    """Case 7: Independently verify C0 forgetting formulas on Official Seed 0 and Our Seed 42 matrices."""
    official_matrix = [
        [0.966],
        [0.949, 0.966],
        [0.935, 0.94, 0.945],
        [0.925, 0.922, 0.937, 0.902],
        [0.912, 0.919, 0.926, 0.896, 0.936],
        [0.893, 0.91, 0.924, 0.877, 0.927, 0.849],
        [0.885, 0.905, 0.92, 0.874, 0.918, 0.839, 0.881],
        [0.867, 0.901, 0.911, 0.861, 0.908, 0.815, 0.872, 0.913],
        [0.866, 0.879, 0.908, 0.855, 0.893, 0.79, 0.872, 0.904, 0.868],
        [0.861, 0.869, 0.903, 0.842, 0.889, 0.758, 0.866, 0.901, 0.872, 0.823],
    ]
    our_matrix = [
        [0.896],
        [0.848, 0.912],
        [0.821, 0.885, 0.846],
        [0.786, 0.874, 0.789, 0.815],
        [0.766, 0.844, 0.766, 0.801, 0.778],
        [0.731, 0.837, 0.767, 0.788, 0.739, 0.755],
        [0.706, 0.821, 0.757, 0.777, 0.732, 0.736, 0.775],
        [0.674, 0.809, 0.755, 0.743, 0.727, 0.717, 0.759, 0.737],
        [0.647, 0.786, 0.748, 0.738, 0.713, 0.711, 0.756, 0.716, 0.693],
        [0.605, 0.762, 0.74, 0.735, 0.705, 0.695, 0.742, 0.71, 0.684, 0.714],
    ]

    # Verify Official Seed 0 (3eca8232)
    off_summary = summary_from_task_matrix(official_matrix)
    assert off_summary['average_incremental_accuracy'] == pytest.approx(0.9098279365079364, abs=1e-12)
    assert off_summary['last_accuracy'] == pytest.approx(0.8584, abs=1e-12)
    # Past-task definition: 0.465 / 9 = 0.051666666666666666 (5.1667%), with Task 9 = -0.004
    assert off_summary['forgetting_past_tasks'] == pytest.approx(0.465 / 9, abs=1e-12)
    assert off_summary['task_forgetting_past_tasks'][8] == pytest.approx(-0.004, abs=1e-12)
    # Official-code definition: 0.469 / 10 = 0.0469 (4.6900%), with Task 9 clamped to 0.0 and Task 10 = 0.0
    assert off_summary['forgetting_official_code'] == pytest.approx(0.0469, abs=1e-12)
    assert off_summary['task_forgetting_official_code'][8] == pytest.approx(0.0, abs=1e-12)
    assert off_summary['task_forgetting_official_code'][9] == pytest.approx(0.0, abs=1e-12)

    # Verify Our Seed 42 (2cdeiptv)
    our_summary = summary_from_task_matrix(our_matrix)
    assert our_summary['average_incremental_accuracy'] == pytest.approx(0.7933317063492064, abs=1e-12)
    assert our_summary['last_accuracy'] == pytest.approx(0.7092, abs=1e-12)
    # Past-task definition: 0.829 / 9 = 0.09211111111111112 (9.2111%)
    assert our_summary['forgetting_past_tasks'] == pytest.approx(0.829 / 9, abs=1e-12)
    # Official-code definition: 0.829 / 10 = 0.0829 (8.2900%)
    assert our_summary['forgetting_official_code'] == pytest.approx(0.0829, abs=1e-12)

    # Also cross-check against stored files in results/ if present
    off_path = Path('results/official_foal_cifar100_seed0/metrics.json')
    if off_path.exists():
        stored_off = json.loads(off_path.read_text(encoding='utf-8'))
        assert stored_off['task_matrix'] == official_matrix
    our_path = Path('results/foal_full_cifar100_seed42/metrics.json')
    if our_path.exists():
        stored_our = json.loads(our_path.read_text(encoding='utf-8'))
        assert stored_our['task_matrix'] == our_matrix


def test_validation_errors():
    """Case 6: Invalid or incomplete accuracy matrices must raise ValueError."""
    with pytest.raises(ValueError, match='empty'):
        validate_task_matrix([])

    with pytest.raises(ValueError, match='expected 2 task accuracies, got 1'):
        validate_task_matrix([[0.5], [0.5]])  # Incomplete row 2

    with pytest.raises(ValueError, match='expected 3 task accuracies, got 2'):
        validate_task_matrix([[0.9], [0.8, 0.85], [0.7, 0.75]])  # Incomplete row 3

    with pytest.raises(ValueError, match='Accuracy must be in range'):
        validate_task_matrix([[1.5]])  # > 1.0

    with pytest.raises(ValueError, match='Accuracy must be in range'):
        validate_task_matrix([[-0.1]])  # < 0.0

    with pytest.raises(ValueError, match='Invalid value'):
        validate_task_matrix([[float('nan')]])

    with pytest.raises(ValueError, match='Invalid value'):
        validate_task_matrix([[float('inf')]])


def test_format_percentage():
    assert format_percentage(0.775) == '77.50%'
    assert format_percentage(0.091234, decimals=3) == '9.123%'
    formatted = format_summary_percentages({
        'average_incremental_accuracy': 0.775,
        'last_accuracy': 0.77,
        'forgetting': 0.09,
        'forgetting_past_tasks': 0.09,
        'forgetting_official_code': 0.045,
    })
    assert formatted['average_incremental_accuracy'] == '77.50%'
    assert formatted['last_accuracy'] == '77.00%'
    assert formatted['forgetting'] == '9.00%'
    assert formatted['forgetting_past_tasks'] == '9.00%'
    assert formatted['forgetting_official_code'] == '4.50%'


def test_aggregate_multi_seed_metrics():
    # 3 seeds
    run1 = {
        'average_incremental_accuracy': 0.78,
        'last_accuracy': 0.75,
        'forgetting': 0.08,
        'forgetting_past_tasks': 0.08,
        'forgetting_official_code': 0.072,
    }
    run2 = {
        'average_incremental_accuracy': 0.80,
        'last_accuracy': 0.77,
        'forgetting': 0.06,
        'forgetting_past_tasks': 0.06,
        'forgetting_official_code': 0.054,
    }
    run3 = {
        'average_incremental_accuracy': 0.82,
        'last_accuracy': 0.79,
        'forgetting': 0.07,
        'forgetting_past_tasks': 0.07,
        'forgetting_official_code': 0.063,
    }

    agg = aggregate_multi_seed_metrics([run1, run2, run3])
    assert agg['num_seeds'] == 3
    assert agg['average_incremental_accuracy_mean'] == pytest.approx(0.80)
    assert agg['average_incremental_accuracy_std'] == pytest.approx(0.02)
    assert agg['last_accuracy_mean'] == pytest.approx(0.77)
    assert agg['last_accuracy_std'] == pytest.approx(0.02)
    assert agg['forgetting_mean'] == pytest.approx(0.07)
    assert agg['forgetting_std'] == pytest.approx(0.01)
    assert agg['forgetting_past_tasks_mean'] == pytest.approx(0.07)
    assert agg['forgetting_official_code_mean'] == pytest.approx(0.063)
    assert agg['average_incremental_accuracy_display'] == '80.00 +/- 2.00% (n=3)'

    # Single seed (n=1): sample std must be None / N/A, never +/- 0.00%
    agg_single = aggregate_multi_seed_metrics([run1])
    assert agg_single['num_seeds'] == 1
    assert agg_single['average_incremental_accuracy_mean'] == pytest.approx(0.78)
    assert agg_single['average_incremental_accuracy_std'] is None
    assert agg_single['average_incremental_accuracy_display'] == '78.00% (n=1, std=N/A)'
    assert '+/- 0.00%' not in agg_single['average_incremental_accuracy_display']
