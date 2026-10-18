import numpy as np
import pytest

from crediwise import evaluate


def test_ks_is_one_for_perfect_separation():
    assert evaluate.ks_statistic(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0


def test_calibration_error_near_zero_when_probabilities_are_honest():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert evaluate.expected_calibration_error(y, p) < 0.01


def test_decision_metrics():
    y = np.array([0, 0, 1, 1])
    d = np.array(["approve", "decline", "approve", "refer"])
    m = evaluate.decision_metrics(y, d)
    assert m["approved_share"] == 0.5
    assert m["default_rate_among_approved"] == 0.5
    assert m["defaulters_not_approved"] == 0.5
    assert m["payers_declined"] == 0.5


def test_score_metrics_compare_against_no_skill():
    y = np.array([0, 1] * 50)
    m = evaluate.score_metrics(y, np.where(y == 1, 0.8, 0.2))
    assert m["auc"] == 1.0
    assert m["brier"] < m["brier_baseline"]
