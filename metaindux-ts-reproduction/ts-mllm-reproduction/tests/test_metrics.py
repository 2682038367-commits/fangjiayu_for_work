import numpy as np

from ts_mllm.metrics import nasa_score, regression_metrics


def test_perfect_predictions_have_zero_error() -> None:
    values = np.asarray([0.0, 10.0, 125.0])
    metrics = regression_metrics(values, values)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["score"] == 0.0


def test_nasa_score_penalizes_late_prediction_more() -> None:
    target = np.asarray([50.0])
    early = nasa_score(target, np.asarray([40.0]))
    late = nasa_score(target, np.asarray([60.0]))
    assert late > early > 0.0
