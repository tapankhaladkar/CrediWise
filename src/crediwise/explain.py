"""Explanations and reasons for declines.

US lending rules (Regulation B, 12 CFR 1002.9) require a lender that turns
an applicant down to state the principal reasons. The model's probability
alone cannot do that, so each decline carries the features that pushed that
customer's risk up the most, measured with SHAP.

SHAP values here are exact and additive: for every customer, the baseline
plus the per-feature contributions equals the model's output. These are
statements about what drove this model's score, not a legally reviewed
adverse-action notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config as cfg

_MONTH_NAMES = {"sep": "September", "aug": "August", "jul": "July",
                "jun": "June", "may": "May", "apr": "April"}

REASONS = {
    "credit_limit": "Size of the credit limit",
    "utilization": "Share of the credit limit in use on the latest statement",
    "avg_utilization": "Average share of the credit limit in use over six months",
    "months_late": "Number of months paid late in the last six",
    "worst_status": "Most serious payment delay in the last six months",
    "paid_share": "Share of the latest bill that was paid",
    **{f"status_{m}": f"Repayment status in {name}" for m, name in _MONTH_NAMES.items()},
    **{f"bill_{m}": f"Bill amount in {name}" for m, name in _MONTH_NAMES.items()},
    **{f"paid_{m}": f"Amount paid in {name}" for m, name in _MONTH_NAMES.items()},
}


class Explainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature push toward default, in log-odds. Shape (rows, features)."""
        return np.asarray(self.explainer.shap_values(X[cfg.FEATURES]))

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        values = np.abs(self.contributions(X)).mean(axis=0)
        return (pd.DataFrame({"feature": cfg.FEATURES, "mean_abs_shap": values})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True))

    def reasons(self, X: pd.DataFrame, top_n: int = 4) -> list[list[dict]]:
        """The features that raised each customer's risk the most."""
        values = self.contributions(X)
        out = []
        for row in values:
            order = np.argsort(row)[::-1][:top_n]
            out.append([
                {"feature": cfg.FEATURES[i], "reason": REASONS[cfg.FEATURES[i]],
                 "contribution": round(float(row[i]), 4)}
                for i in order if row[i] > 0
            ])
        return out
