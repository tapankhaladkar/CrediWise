"""Fairness audit.

SCOPE LIMITATION -- read this before quoting any number from this module.

ECOA / Regulation B name seven protected bases. This dataset contains exactly
one of them:

    Race / colour                 ABSENT
    Religion                      ABSENT
    National origin               ABSENT
    Sex                           ABSENT
    Marital status                ABSENT
    Receipt of public assistance  ABSENT
    Age                           PRESENT (8.5% of values corrupt)

and only 5 rows fall in the 62+ bracket that ECOA specifically protects, so the
classic age test is statistically empty. What this module can honestly measure
is outcome disparity across age bands, occupation and income quintiles. It
cannot clear a model of race or sex discrimination, and nothing here should be
presented as if it could. See docs/FAIRNESS.md.

Note also that `Age` is excluded from the model's inputs, which does *not* make
the model age-neutral: the labels themselves are age-correlated (16% of the
18-45 bands are "Good" against 33% of the 46-55 band), so disparity can and
does arrive through proxies. That is exactly why outcomes are measured here
rather than inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .evaluate import expected_calibration_error

FOUR_FIFTHS = 0.8


def _rates(y_true: np.ndarray, y_pred: np.ndarray, favourable: int) -> dict:
    selected = y_pred >= favourable
    deserving = y_true >= favourable
    tpr = float(selected[deserving].mean()) if deserving.any() else float("nan")
    fpr = float(selected[~deserving].mean()) if (~deserving).any() else float("nan")
    return {
        "n": int(len(y_true)),
        "selection_rate": float(selected.mean()),
        "base_rate": float(deserving.mean()),
        "tpr": tpr,
        "fpr": fpr,
    }


def group_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    groups: pd.Series,
    favourable: int = 2,
    min_group_size: int = 100,
) -> pd.DataFrame:
    """Per-subgroup selection, error and calibration rates.

    ``favourable`` is the lowest predicted class counted as a favourable
    outcome -- 2 ("Good") by default.
    """
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(axis=1)
    rows = []
    for value, mask in groups.groupby(groups, observed=True).groups.items():
        idx = groups.index.get_indexer(mask)
        if len(idx) < min_group_size:
            continue
        stats = _rates(y_true[idx], y_pred[idx], favourable)
        stats["group"] = str(value)
        stats["accuracy"] = float((y_true[idx] == y_pred[idx]).mean())
        stats["ece"] = expected_calibration_error(y_true[idx], proba[idx])
        rows.append(stats)
    cols = ["group", "n", "base_rate", "selection_rate", "tpr", "fpr",
            "accuracy", "ece"]
    return pd.DataFrame(rows)[cols].sort_values("group").reset_index(drop=True)


def disparity_summary(report: pd.DataFrame) -> dict:
    """Collapse a per-group report into the headline disparity measures."""
    if report.empty:
        return {"status": "no groups above minimum size"}
    rates = report["selection_rate"]
    ratio = float(rates.min() / rates.max()) if rates.max() > 0 else float("nan")
    return {
        "disparate_impact_ratio": ratio,
        "passes_four_fifths_rule": bool(ratio >= FOUR_FIFTHS),
        "selection_rate_range": [float(rates.min()), float(rates.max())],
        "equal_opportunity_gap": float(report["tpr"].max() - report["tpr"].min()),
        "equalised_odds_gap": float(
            max(report["tpr"].max() - report["tpr"].min(),
                report["fpr"].max() - report["fpr"].min())
        ),
        "calibration_gap": float(report["ece"].max() - report["ece"].min()),
        "worst_group": str(report.loc[rates.idxmin(), "group"]),
        "best_group": str(report.loc[rates.idxmax(), "group"]),
    }


def audit(
    y_true: np.ndarray,
    proba: np.ndarray,
    audit_frame: pd.DataFrame,
    dimensions: list[str] | None = None,
    favourable: int = 2,
) -> dict:
    """Run the audit across every available dimension."""
    results = {}
    for dimension in dimensions or cfg.AUDIT_DIMENSIONS:
        if dimension not in audit_frame.columns:
            continue
        series = audit_frame[dimension].astype("object").fillna("Unknown")
        report = group_report(y_true, proba, series, favourable=favourable)
        results[dimension] = {
            "by_group": report.to_dict(orient="records"),
            "summary": disparity_summary(report),
        }
    results["_scope"] = {
        "testable_ecoa_bases": ["Age (via age_band)"],
        "untestable_ecoa_bases": [
            "Race/colour", "Religion", "National origin", "Sex",
            "Marital status", "Receipt of public assistance",
        ],
        "caveat": (
            "Six of seven ECOA protected bases are absent from this dataset. "
            "This audit cannot clear the model on them."
        ),
    }
    return results
