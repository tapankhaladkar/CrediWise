import numpy as np
import pandas as pd

from crediwise import fairness


def _proba_from_labels(labels, n_classes=3):
    proba = np.full((len(labels), n_classes), 0.05)
    for i, k in enumerate(labels):
        proba[i, k] = 0.90
    return proba / proba.sum(axis=1, keepdims=True)


def test_equal_groups_pass_four_fifths():
    # Both groups get the same mix of outcomes, so selection rates match.
    y = np.array([2, 0, 2, 0] * 100)
    proba = _proba_from_labels(y)
    groups = pd.Series(["A", "A", "B", "B"] * 100)
    report = fairness.group_report(y, proba, groups)
    summary = fairness.disparity_summary(report)
    assert summary["disparate_impact_ratio"] == 1.0
    assert summary["passes_four_fifths_rule"]


def test_skewed_selection_fails_four_fifths():
    y = np.concatenate([np.full(200, 2), np.full(200, 0)])
    proba = _proba_from_labels(y)
    groups = pd.Series(["A"] * 200 + ["B"] * 200)
    report = fairness.group_report(y, proba, groups)
    summary = fairness.disparity_summary(report)
    assert summary["disparate_impact_ratio"] < 0.8
    assert not summary["passes_four_fifths_rule"]


def test_audit_always_declares_scope_limits():
    """The report must never imply it cleared race or sex."""
    y = np.array([0, 1, 2] * 100)
    proba = _proba_from_labels(y)
    frame = pd.DataFrame({"age_band": ["18-25", "26-35", "36-45"] * 100})
    result = fairness.audit(y, proba, frame, dimensions=["age_band"])
    scope = result["_scope"]
    assert "Sex" in scope["untestable_ecoa_bases"]
    assert "Race/colour" in scope["untestable_ecoa_bases"]
    assert len(scope["untestable_ecoa_bases"]) == 6


def test_small_groups_excluded():
    y = np.array([0] * 500 + [2] * 5)
    proba = _proba_from_labels(y)
    groups = pd.Series(["big"] * 500 + ["tiny"] * 5)
    report = fairness.group_report(y, proba, groups, min_group_size=100)
    assert "tiny" not in set(report["group"])


def test_unknown_bucket_excluded_from_ratio():
    """'Unknown' is a missing-data bucket, not a protected class.

    Letting it set the minimum would report a disparity against people who
    do not form a group.
    """
    y = np.array([2] * 300 + [0] * 300)
    proba = _proba_from_labels(y)
    groups = pd.Series(["A"] * 300 + ["Unknown"] * 300)
    summary = fairness.disparity_summary(
        fairness.group_report(y, proba, groups)
    )
    assert summary["excluded_from_ratio"] == ["Unknown"]
    assert summary["disparate_impact_ratio"] == 1.0, \
        "with only 'A' scored, the ratio should be 1.0, not 0.0"


def test_amplification_detects_model_widening_label_disparity():
    """amplification < 1 means the model widened what the labels already had."""
    # Group A: 50% truly Good, model selects all of them.
    # Group B: 50% truly Good, model selects none -> base rates equal,
    # selection rates maximally unequal.
    y = np.array([2, 0] * 100 + [2, 0] * 100)
    proba = _proba_from_labels(np.array([2, 0] * 100 + [0, 0] * 100))
    groups = pd.Series(["A"] * 200 + ["B"] * 200)
    summary = fairness.disparity_summary(fairness.group_report(y, proba, groups))
    assert summary["base_rate_ratio"] == 1.0
    assert summary["amplification"] < 1.0
