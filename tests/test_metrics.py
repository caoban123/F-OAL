import pytest
from src.foal.metrics import summary_from_task_matrix


def test_incremental_summary_values():
    summary = summary_from_task_matrix([[0.9], [0.8, 0.7], [0.7, 0.65, 0.6]])
    assert summary['average_incremental_accuracy'] == pytest.approx((0.9+0.75+0.65)/3)
    assert summary['last_accuracy'] == pytest.approx(0.65)
    assert summary['forgetting'] == pytest.approx(((0.9-0.7)+(0.7-0.65))/2)
