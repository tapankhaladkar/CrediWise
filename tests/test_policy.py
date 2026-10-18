import pytest

from crediwise.policy import DecisionPolicy


def test_approve_threshold_follows_from_costs():
    """Default costs 5x a lost customer -> approve while p < 1/6."""
    assert DecisionPolicy().approve_below == pytest.approx(1 / 6)
    assert DecisionPolicy(cost_default=3, cost_lost_customer=1).approve_below == pytest.approx(0.25)


def test_three_bands():
    decisions = DecisionPolicy().decide([0.05, 0.166, 0.167, 0.49, 0.5, 0.95])
    assert list(decisions) == ["approve", "approve", "refer", "refer", "decline", "decline"]


def test_inconsistent_thresholds_are_rejected():
    with pytest.raises(ValueError):
        DecisionPolicy(cost_default=1, cost_lost_customer=5, decline_at=0.5)
