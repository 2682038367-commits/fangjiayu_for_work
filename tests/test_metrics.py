import numpy as np

from rul_chronos.metrics import phm_score, rmse


def test_metrics_are_zero_for_exact_prediction():
    target = np.array([1.0, 2.0, 3.0])
    assert rmse(target, target) == 0.0
    assert phm_score(target, target) == 0.0


def test_late_prediction_is_penalized_more():
    target = np.array([50.0])
    assert phm_score(target, np.array([60.0])) > phm_score(target, np.array([40.0]))

