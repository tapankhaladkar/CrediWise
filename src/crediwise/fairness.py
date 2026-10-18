"""Fairness audit.

The model never sees sex, marital status, age or education. That does not
make it fair: other features can stand in for them, and the outcomes can
still differ between groups. So the audit measures outcomes, group by group.

What it measures, per group:
- approval rate, and the ratio of the lowest to the highest (the
  "four-fifths rule" flags anything under 0.8);
- approval rate among customers who actually repaid -- whether creditworthy
  people are treated alike (equal opportunity);
- whether the predicted default rate matches the actual one (calibration).

It also compares the approval gap with the gap in actual repayment rates.
Groups that genuinely repay at different rates will be approved at different
rates; the useful question is whether the model widens or narrows that gap.

Scope: the data covers three of the seven bases protected under US
fair-lending law (sex, marital status, age). Race or colour, religion,
national origin and receipt of public assistance are not in the data and
cannot be tested. The data is from Taiwan; US rules are applied as a
benchmark, not because they govern it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.8
MIN_GROUP_SIZE = 100
EXCLUDED_GROUPS = ("unknown",)


def group_report(y, p, decisions, groups: pd.Series,
                 min_group_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    y, p, d = np.asarray(y), np.asarray(p), np.asarray(decisions)
    g = np.asarray(groups.astype("object").fillna("unknown"))
    rows = []
    for value in pd.unique(g):
        mask = g == value
        if mask.sum() < min_group_size:
            continue
        approved = d[mask] == "approve"
        repaid = y[mask] == 0
        rows.append({
            "group": str(value),
            "n": int(mask.sum()),
            "default_rate": float(y[mask].mean()),
            "predicted_default_rate": float(p[mask].mean()),
            "approval_rate": float(approved.mean()),
            "approval_rate_if_repaid": float(approved[repaid].mean()),
            "default_rate_if_approved": float(y[mask][approved].mean()) if approved.any() else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def disparity_summary(report: pd.DataFrame, exclude=EXCLUDED_GROUPS) -> dict:
    """Headline measures, ignoring missing-data buckets such as "unknown"."""
    scored = report[~report["group"].isin(exclude)]
    if len(scored) < 2:
        return {"status": "fewer than two groups large enough to compare"}
    approval = scored["approval_rate"]
    repay = 1 - scored["default_rate"]
    ratio = float(approval.min() / approval.max())
    repay_ratio = float(repay.min() / repay.max())
    return {
        "approval_ratio": ratio,
        "passes_four_fifths": bool(ratio >= FOUR_FIFTHS),
        "repayment_ratio": repay_ratio,
        # Below 1: approvals are more unequal than actual repayment.
        "amplification": float(ratio / repay_ratio),
        "lowest_approval": str(scored.loc[approval.idxmin(), "group"]),
        "highest_approval": str(scored.loc[approval.idxmax(), "group"]),
        "equal_opportunity_gap": float(scored["approval_rate_if_repaid"].max()
                                       - scored["approval_rate_if_repaid"].min()),
        "max_calibration_gap": float((scored["predicted_default_rate"]
                                      - scored["default_rate"]).abs().max()),
        "excluded": [g for g in exclude if g in set(report["group"])],
    }


def audit(y, p, decisions, frame: pd.DataFrame, dimensions) -> dict:
    results = {}
    for dim in dimensions:
        column = frame[dim].reset_index(drop=True)
        report = group_report(y, p, decisions, column)
        counts = column.astype("object").fillna("unknown").value_counts()
        results[dim] = {
            "by_group": report.to_dict(orient="records"),
            "summary": disparity_summary(report),
            # Listed rather than silently dropped: these groups exist but are
            # too small in the test set to measure reliably.
            "too_small_to_assess": {str(k): int(v) for k, v in counts.items()
                                    if v < MIN_GROUP_SIZE},
        }
    results["_scope"] = {
        "testable": ["sex", "marital status", "age"],
        "not_in_data": ["race or colour", "religion", "national origin",
                        "receipt of public assistance"],
    }
    return results
