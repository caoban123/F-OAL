"""Incremental evaluation metrics; validate paper-specific definitions separately."""
from __future__ import annotations


def summary_from_task_matrix(matrix: list[list[float]]) -> dict[str, float]:
    """Rows are stages; columns are task accuracies known at that stage.

    Returns task-averaged stage mean ('average_incremental_accuracy'), final
    stage mean, and a common maximum-minus-final task forgetting definition.
    The paper may use a distinct F metric: cite and verify before comparison.
    """
    if not matrix or any(len(row) != i + 1 for i, row in enumerate(matrix)):
        raise ValueError('Expected triangular task accuracy matrix')
    means = [sum(row) / len(row) for row in matrix]
    old_tasks = len(matrix) - 1
    forgetting = (
        sum(max(matrix[t][j] for t in range(j, len(matrix)-1)) - matrix[-1][j]
            for j in range(old_tasks)) / old_tasks
        if old_tasks else 0.0
    )
    return dict(average_incremental_accuracy=sum(means) / len(means),
                last_accuracy=means[-1], forgetting=forgetting)
