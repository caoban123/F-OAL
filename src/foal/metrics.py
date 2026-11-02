"""Continual learning evaluation metrics (C0 Metric Alignment Verified).

Formal Mathematical Definitions:
Following standard continual learning formulations (Lopez-Paz & Ranzato, 2017;
Chaudhry et al., 2018), the F-OAL paper (NeurIPS 2024, Section 4.2, Equations 16-18),
and the official author repository (liuyuchen-cz/F-OAL @ 73e4ac1, experiment/metrics.py):

Let T (denoted m in paper Section 4.2) be the number of sequential tasks.
Let a_{k, j} in [0, 1] denote the test accuracy on task j after training on task k (where 1 <= j <= k <= T).
The accuracy history is represented as a lower-triangular matrix R where R[k-1][j-1] = a_{k, j}.

1. Stage Accuracy (A_k, Paper Equation 16):
   Mean accuracy over all seen tasks {1, ..., k} after completing stage k:
       A_k = (1 / k) * sum_{j=1}^k a_{k, j}

2. Average Incremental Accuracy (bar{A} or A_{avg}):
   Average of stage accuracies across all stages 1 to T:
       bar{A} = (1 / T) * sum_{k=1}^T A_k = (1 / T) * sum_{k=1}^T (1 / k) * sum_{j=1}^k a_{k, j}

3. Last Accuracy (A_T or A_{last}):
   Final accuracy over all seen tasks after completing all T tasks:
       A_T = (1 / T) * sum_{j=1}^T a_{T, j}

4. Forgetting — Paper-Style / Past-Task Definition (forgetting_past_tasks, Paper Equations 17 & 18):
   In F-OAL Section 4.2, Equation (18) defines per-task forgetting for each past task j < k as:
       f_{T, j}^{past} = max_{l in {j, ..., T-1}} a_{l, j} - a_{T, j},    for j in {1, ..., T-1}
   and Equation (17) averages strictly over the T - 1 past tasks:
       F_T^{past} = (1 / (T - 1)) * sum_{j=1}^{T-1} f_{T, j}^{past}
   (When T = 1, F_1^{past} = 0.0 by definition.)
   Note: Under this definition, if a past task's final accuracy exceeds its historical peak prior
   to stage T (positive backward transfer, a_{T, j} > max_{l < T} a_{l, j}), then f_{T, j}^{past} < 0
   unless clip_zero=True is explicitly requested.

5. Forgetting — Official Author Code Definition (forgetting_official_code, experiment/metrics.py:18,22-26 & 49-54):
   In the official repository (`experiment/metrics.py`, functions `compute_performance` and
   `single_run_avg_end_fgt`), forgetting is computed across all T tasks (including the final task T)
   and includes the final stage k = T inside the maximum:
       f_{T, j}^{code} = max_{l in {j, ..., T}} a_{l, j} - a_{T, j},      for j in {1, ..., T}
       F_T^{code} = (1 / T) * sum_{j=1}^T f_{T, j}^{code}
   Two mathematical consequences distinguish F_T^{code} from F_T^{past}:
   (a) Including l = T in the maximum clamps any negative forgetting to 0.0 (f_{T, j}^{code} >= 0)
       and forces the final task j = T to have f_{T, T}^{code} = a_{T, T} - a_{T, T} = 0.0.
   (b) Averaging over all T tasks divides by T rather than T - 1 (scaling by (T - 1) / T = 0.9 for T = 10
       whenever all past tasks have non-negative forgetting).
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence


def validate_task_matrix(matrix: Sequence[Sequence[float]]) -> None:
    """Validate that the matrix is non-empty, lower-triangular, and contains valid ratios."""
    if not matrix:
        raise ValueError('Task matrix cannot be empty')
    for stage_idx, row in enumerate(matrix):
        expected_len = stage_idx + 1
        if len(row) != expected_len:
            raise ValueError(f'Stage {stage_idx + 1} expected {expected_len} task accuracies, got {len(row)}')
        for task_idx, acc in enumerate(row):
            if not isinstance(acc, (int, float)) or isinstance(acc, bool) or math.isnan(acc) or math.isinf(acc):
                raise ValueError(f'Invalid value at stage {stage_idx + 1}, task {task_idx + 1}: {acc}')
            if not (0.0 <= acc <= 1.0):
                raise ValueError(f'Accuracy must be in range [0, 1], got {acc} at stage {stage_idx + 1}, task {task_idx + 1}')


def stage_accuracies(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Compute A_k for each stage k in 1..T."""
    validate_task_matrix(matrix)
    return [sum(row) / len(row) for row in matrix]


def task_forgetting_past_tasks(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> list[float]:
    """Compute paper-style per-task forgetting f_{T, j}^{past} for each past task j in 1..T-1 (Eq. 18)."""
    validate_task_matrix(matrix)
    num_stages = len(matrix)
    if num_stages <= 1:
        return []

    past_tasks = num_stages - 1
    forgetting_values: list[float] = []
    for j in range(past_tasks):
        peak_acc = max(matrix[t][j] for t in range(j, num_stages - 1))
        final_acc = matrix[-1][j]
        f_j = peak_acc - final_acc
        if clip_zero:
            f_j = max(0.0, f_j)
        forgetting_values.append(float(f_j))
    return forgetting_values


def forgetting_past_tasks(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> float:
    """Compute paper-style average forgetting F_T^{past} across all T-1 past tasks (Eq. 17)."""
    f_list = task_forgetting_past_tasks(matrix, clip_zero=clip_zero)
    if not f_list:
        return 0.0
    return float(sum(f_list) / len(f_list))


def task_forgetting_official_code(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Compute official-code per-task forgetting f_{T, j}^{code} for j in 1..T (experiment/metrics.py:23-24)."""
    validate_task_matrix(matrix)
    num_stages = len(matrix)
    forgetting_values: list[float] = []
    for j in range(num_stages):
        peak_acc = max(matrix[t][j] for t in range(j, num_stages))
        final_acc = matrix[-1][j]
        f_j = peak_acc - final_acc
        forgetting_values.append(float(f_j))
    return forgetting_values


def forgetting_official_code(matrix: Sequence[Sequence[float]]) -> float:
    """Compute official-code average forgetting F_T^{code} across all T tasks (experiment/metrics.py:23-25)."""
    f_list = task_forgetting_official_code(matrix)
    if not f_list:
        return 0.0
    return float(sum(f_list) / len(f_list))


# Backwards-compatible aliases
def task_forgetting(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> list[float]:
    """Compute forgetting f_j for each past task j in 1..T-1 (alias for task_forgetting_past_tasks)."""
    return task_forgetting_past_tasks(matrix, clip_zero=clip_zero)


def average_forgetting(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> float:
    """Compute average forgetting F_T across all past tasks (alias for forgetting_past_tasks)."""
    return forgetting_past_tasks(matrix, clip_zero=clip_zero)


def summary_from_task_matrix(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> dict[str, Any]:
    """Calculate all standard continual learning metrics from a task accuracy matrix.

    Exposes both `forgetting_past_tasks` (Paper Eq. 17-18, T-1 denominator) and
    `forgetting_official_code` (official repo `experiment/metrics.py`, T denominator with
    current-stage max), while keeping `forgetting` and `task_forgetting` backwards-compatible.
    """
    validate_task_matrix(matrix)
    stages = stage_accuracies(matrix)
    avg_inc_acc = sum(stages) / len(stages)
    last_acc = stages[-1]
    f_list_past = task_forgetting_past_tasks(matrix, clip_zero=clip_zero)
    avg_f_past = sum(f_list_past) / len(f_list_past) if f_list_past else 0.0
    f_list_code = task_forgetting_official_code(matrix)
    avg_f_code = sum(f_list_code) / len(f_list_code) if f_list_code else 0.0

    return {
        'average_incremental_accuracy': float(avg_inc_acc),
        'last_accuracy': float(last_acc),
        'forgetting': float(avg_f_past),
        'forgetting_past_tasks': float(avg_f_past),
        'forgetting_official_code': float(avg_f_code),
        'num_tasks': len(matrix),
        'stage_accuracies': [float(s) for s in stages],
        'task_forgetting': [float(f) for f in f_list_past],
        'task_forgetting_past_tasks': [float(f) for f in f_list_past],
        'task_forgetting_official_code': [float(f) for f in f_list_code],
        'final_task_accuracies': [float(acc) for acc in matrix[-1]],
    }


def format_percentage(ratio: float, decimals: int = 2) -> str:
    """Format a ratio in [0, 1] as a percentage string 'xx.xx%'."""
    return f'{ratio * 100:.{decimals}f}%'


def format_summary_percentages(summary: dict[str, Any], decimals: int = 2) -> dict[str, str]:
    """Convert float ratio metrics into formatted percentage strings for reporting."""
    formatted = {}
    for key in [
        'average_incremental_accuracy',
        'last_accuracy',
        'forgetting',
        'forgetting_past_tasks',
        'forgetting_official_code',
    ]:
        if key in summary and isinstance(summary[key], (int, float)):
            formatted[key] = format_percentage(summary[key], decimals=decimals)
    return formatted


def aggregate_multi_seed_metrics(runs_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric dictionaries across multiple independent seed runs.

    Computes sample mean and sample standard deviation (Bessel's correction, N-1).
    When N == 1, sample standard deviation is undefined (None / 'N/A').
    """
    if not runs_metrics:
        raise ValueError('No run metrics provided for aggregation')

    n = len(runs_metrics)
    metric_keys = [
        'average_incremental_accuracy',
        'last_accuracy',
        'forgetting',
        'forgetting_past_tasks',
        'forgetting_official_code',
    ]
    aggregated: dict[str, Any] = {'num_seeds': n}

    for key in metric_keys:
        values = [run[key] for run in runs_metrics if key in run and run[key] is not None]
        if len(values) != n:
            continue
        mean_val = sum(values) / n
        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
            std_val: float | None = float(math.sqrt(variance))
            display_str = f'{mean_val * 100:.2f} +/- {std_val * 100:.2f}% (n={n})'
        else:
            std_val = None
            display_str = f'{mean_val * 100:.2f}% (n=1, std=N/A)'

        aggregated[f'{key}_mean'] = float(mean_val)
        aggregated[f'{key}_std'] = std_val
        aggregated[f'{key}_display'] = display_str

    return aggregated
