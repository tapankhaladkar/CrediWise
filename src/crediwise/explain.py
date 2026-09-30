"""Explainability and ECOA reason codes.

Regulation B (12 CFR 1002.9) requires a creditor taking adverse action to give
the *principal reasons* for it. A probability alone does not satisfy that, so
the scoring path returns per-decision attributions derived from SHAP values on
the underlying forest.

These are statements about what drove *this model's* output. They are not a
legal adverse-action notice, and a deployment would need counsel to sign off on
the wording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from . import config as cfg

#: Plain-language stems for the reasons a decision can cite. Keyed by the
#: encoded feature name; one-hot columns fall back to a generic phrasing.
REASON_TEMPLATES = {
    "Outstanding_Debt": "Level of outstanding debt",
    "Interest_Rate": "Interest rate on existing credit",
    "Credit_Mix": "Mix of credit types held",
    "Delay_from_due_date": "Average days past due on payments",
    "Changed_Credit_Limit": "Recent change in credit limit",
    "Num_Credit_Inquiries": "Number of recent credit inquiries",
    "Credit_Utilization_Ratio": "Proportion of available credit in use",
    "Num_of_Delayed_Payment": "Number of delayed payments",
    "Credit_History_Months": "Length of credit history",
    "Annual_Income": "Annual income relative to obligations",
    "Monthly_Inhand_Salary": "Monthly take-home pay",
    "Monthly_Balance": "Monthly balance retained",
    "Amount_invested_monthly": "Monthly amount invested",
    "Num_Bank_Accounts": "Number of bank accounts held",
    "Num_Credit_Card": "Number of credit cards held",
    "Num_of_Loan": "Number of active loans",
    "Total_EMI_per_month": "Total monthly loan instalments",
    "Payment_of_Min_Amount": "Pattern of paying only the minimum amount",
}


def _describe(encoded_name: str) -> str:
    if encoded_name in REASON_TEMPLATES:
        return REASON_TEMPLATES[encoded_name]
    for base in ("Occupation_", "Payment_Behaviour_"):
        if encoded_name.startswith(base):
            label = encoded_name[len(base):].replace("_", " ").lower()
            field = base.rstrip("_").replace("_", " ").lower()
            return f"Recorded {field} ({label})"
    return encoded_name.replace("_", " ")


class Explainer:
    """Wraps a fitted pipeline with a SHAP TreeExplainer."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame | None = None):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.explainer = shap.TreeExplainer(self.classifier)
        self._background = background

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Returns array shaped (n_samples, n_features, n_classes)."""
        transformed = self.preprocessor.transform(X)
        values = self.explainer.shap_values(transformed, check_additivity=False)
        if isinstance(values, list):  # older shap returns a list per class
            values = np.stack(values, axis=-1)
        return values

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature, averaged over classes."""
        values = self.shap_values(X)
        magnitude = np.abs(values).mean(axis=(0, 2))
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": magnitude})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def reason_codes(
        self, X: pd.DataFrame, top_n: int = 4, adverse_class: int = 0
    ) -> list[list[dict]]:
        """Principal reasons pushing each row toward the adverse band.

        Positive SHAP for the "Poor" class means the feature pushed the
        decision toward the adverse outcome, so the top contributors by that
        measure are the ones a notice would cite.
        """
        values = self.shap_values(X)
        adverse = values[:, :, adverse_class]
        out = []
        for row in range(adverse.shape[0]):
            order = np.argsort(adverse[row])[::-1][:top_n]
            out.append([
                {
                    "feature": self.feature_names[i],
                    "reason": _describe(self.feature_names[i]),
                    "contribution": round(float(adverse[row, i]), 5),
                }
                for i in order
                if adverse[row, i] > 0
            ])
        return out
