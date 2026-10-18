import numpy as np
import pandas as pd

from crediwise import fairness


def _run(y, decisions, groups, p=None):
    y = np.asarray(y)
    p = np.where(y == 1, 0.7, 0.1) if p is None else p
    return fairness.group_report(y, p, np.asarray(decisions), pd.Series(groups))


def test_identical_treatment_passes():
    y = [0, 1] * 200
    d = ["approve", "decline"] * 200
    g = ["A", "A", "B", "B"] * 100
    s = fairness.disparity_summary(_run(y, d, g))
    assert s["approval_ratio"] == 1.0 and s["passes_four_fifths"]


def test_unequal_approval_fails_four_fifths():
    y = [0] * 400
    d = ["approve"] * 200 + ["approve"] * 50 + ["decline"] * 150
    g = ["A"] * 200 + ["B"] * 200
    s = fairness.disparity_summary(_run(y, d, g))
    assert s["approval_ratio"] == 0.25 and not s["passes_four_fifths"]
    assert s["lowest_approval"] == "B"


def test_amplification_separates_model_from_data():
    """Equal repayment but unequal approval: the model widened the gap."""
    y = [0] * 400
    d = ["approve"] * 200 + ["approve"] * 100 + ["decline"] * 100
    g = ["A"] * 200 + ["B"] * 200
    s = fairness.disparity_summary(_run(y, d, g))
    assert s["repayment_ratio"] == 1.0
    assert s["amplification"] < 1


def test_unknown_bucket_is_not_compared():
    y = [0] * 400
    d = ["approve"] * 200 + ["decline"] * 200
    g = ["A"] * 200 + ["unknown"] * 200
    s = fairness.disparity_summary(_run(y, d, g))
    assert s == {"status": "fewer than two groups large enough to compare"}


def test_small_groups_are_listed_not_hidden():
    y = np.zeros(250, dtype=int)
    d = np.array(["approve"] * 250)
    frame = pd.DataFrame({"age_band": ["30-39"] * 120 + ["40-49"] * 120 + ["60-79"] * 10})
    result = fairness.audit(y, np.full(250, 0.1), d, frame, ["age_band"])
    assert result["age_band"]["too_small_to_assess"] == {"60-79": 10}


def test_scope_names_what_cannot_be_tested():
    frame = pd.DataFrame({"sex": ["male", "female"] * 100})
    result = fairness.audit(np.zeros(200, int), np.full(200, .1),
                            np.array(["approve"] * 200), frame, ["sex"])
    assert "race or colour" in result["_scope"]["not_in_data"]
