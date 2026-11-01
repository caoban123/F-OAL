"""Continual learning evaluation metrics.

Formal Mathematical Definitions:
Following standard continual learning formulations (Lopez-Paz & Ranzato, 2017;
Chaudhry et al., 2018) and the F-OAL evaluation protocol (NeurIPS 2024, Section 5):

Let T be the number of sequential tasks.
Let a_{k, j} in [0, 1] denote the test accuracy on task j after training on task k (where k >= j).
The accuracy history is represented as a lower-triangular matrix R where R[k-1][j-1] = a_{k, j}.

1. Stage Accuracy (A_k):
   Mean accuracy over all seen tasks {1, ..., k} after completing stage k:
       A_k = (1 / k) * sum_{j=1}^k a_{k, j}

2. Average Incremental Accuracy (bar{A}):
   Average of stage accuracies across all stages 1 to T:
       bar{A} = (1 / T) * sum_{k=1}^T A_k = (1 / T) * sum_{k=1}^T (1 / k) * sum_{j=1}^k a_{k, j}

3. Last Accuracy (A_T):
   Final accuracy over all seen tasks after completing all T tasks:
       A_T = (1 / T) * sum_{j=1}^T a_{T, j}

4. Task Forgetting (f_j^T):
   For a previous task j in {1, ..., T-1}, the performance drop from its peak historical accuracy:
       f_j^T = max_{k in {j, ..., T-1}} a_{k, j} - a_{T, j}

5. Average Forgetting (F_T):
   Average forgetting across all past tasks (T > 1):
       F_T = (1 / (T - 1)) * sum_{j=1}^{T-1} f_j^T
   When T = 1, F_1 = 0.0 by definition.
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
            if not isinstance(acc, (int, float)) or math.isnan(acc) or math.isinf(acc):
                raise ValueError(f'Invalid value at stage {stage_idx + 1}, task {task_idx + 1}: {acc}')
            if not (0.0 <= acc <= 1.0):
                raise ValueError(f'Accuracy must be in range [0, 1], got {acc} at stage {stage_idx + 1}, task {task_idx + 1}')


def stage_accuracies(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Compute A_k for each stage k in 1..T."""
    validate_task_matrix(matrix)
    return [sum(row) / len(row) for row in matrix]


def task_forgetting(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> list[float]:
    """Compute forgetting f_j for each past task j in 1..T-1."""
    validate_task_matrix(matrix)
    num_stages = len(matrix)
    if num_stages <= 1:
        return []
    
    past_tasks = num_stages - 1
    forgetting_values: list[float] = []
    for j in range(past_tasks):
        # Peak historical accuracy for task j before the final stage
        peak_acc = max(matrix[t][j] for t in range(j, num_stages - 1))
        final_acc = matrix[-1][j]
        f_j = peak_acc - final_acc
        if clip_zero:
            f_j = max(0.0, f_j)
        forgetting_values.append(float(f_j))
    return forgetting_values


def average_forgetting(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> float:
    """Compute average forgetting F_T across all past tasks."""
    f_list = task_forgetting(matrix, clip_zero=clip_zero)
    if not f_list:
        return 0.0
    return sum(f_list) / len(f_list)


def summary_from_task_matrix(matrix: Sequence[Sequence[float]], clip_zero: bool = False) -> dict[str, Any]:
    """Calculate all standard continual learning metrics from a task accuracy matrix.
    
    All returned values are ratios in [0, 1] (or signed forgetting in [-1, 1]).
    """
    validate_task_matrix(matrix)
    stages = stage_accuracies(matrix)
    avg_inc_acc = sum(stages) / len(stages)
    last_acc = stages[-1]
    f_list = task_forgetting(matrix, clip_zero=clip_zero)
    avg_f = sum(f_list) / len(f_list) if f_list else 0.0

    return {
        'average_incremental_accuracy': float(avg_inc_acc),
        'last_accuracy': float(last_acc),
        'forgetting': float(avg_f),
        'num_tasks': len(matrix),
        'stage_accuracies': [float(s) for s in stages],
        'task_forgetting': [float(f) for f in f_list],
        'final_task_accuracies': [float(acc) for acc in matrix[-1]],
    }


def format_percentage(ratio: float, decimals: int = 2) -> str:
    """Format a ratio in [0, 1] as a percentage string 'xx.xx%'."""
    return f'{ratio * 100:.{decimals}f}%'


def format_summary_percentages(summary: dict[str, Any], decimals: int = 2) -> dict[str, str]:
    """Convert float ratio metrics into formatted percentage strings for reporting."""
    formatted = {}
    for key in ['average_incremental_accuracy', 'last_accuracy', 'forgetting']:
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
    metric_keys = ['average_incremental_accuracy', 'last_accuracy', 'forgetting']
    aggregated: dict[str, Any] = {'num_seeds': n}

    for key in metric_keys:
        values = [run[key] for run in runs_metrics if key in run]
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
