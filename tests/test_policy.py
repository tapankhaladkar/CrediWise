import numpy as np

from crediwise.policy import DecisionPolicy, DEFAULT_COST_MATRIX


def test_high_good_probability_approves():
    policy = DecisionPolicy()
    proba = np.array([[0.05, 0.15, 0.80]])
    assert policy.decide(proba)[0]["decision"] == "approve"


def test_low_good_probability_declines_and_is_adverse():
    policy = DecisionPolicy()
    proba = np.array([[0.80, 0.15, 0.05]])
    decision = policy.decide(proba)[0]
    assert decision["decision"] == "decline"
    assert decision["is_adverse"] is True


def test_middle_band_refers():
    policy = DecisionPolicy()
    proba = np.array([[0.30, 0.30, 0.40]])
    assert policy.decide(proba)[0]["decision"] == "refer"


def test_cost_matrix_penalises_approving_poor_applicants():
    """Approving a true Poor must cost more than declining a true Good."""
    assert DEFAULT_COST_MATRIX[0, 2] > DEFAULT_COST_MATRIX[2, 0]


def test_expected_cost_shape():
    policy = DecisionPolicy()
    proba = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    assert policy.expected_cost(proba).shape == (2, 3)
