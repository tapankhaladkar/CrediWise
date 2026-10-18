"""Metrics.

22% of customers default, so a model that always predicts "will repay" is
78% accurate and useless. Accuracy is therefore not reported as a headline.
The model is judged on how well it ranks risk (AUC, KS), how well it finds
defaulters (precision-recall), whether its probabilities can be taken at face
value (Brier, calibration error), and what its decisions do to a loan book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Largest gap between the score distributions of defaulters and payers."""
    order = np.argsort(p)
    y = np.asarray(y)[order]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float(np.max(np.abs(np.cumsum(y) / pos - np.cumsum(1 - y) / neg)))


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. observed default rate per probability bin."""
    bins = pd.cut(p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return (
        pd.DataFrame({"p": p, "y": np.asarray(y), "bin": bins})
        .groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
        .reset_index()
        .assign(bin=lambda t: t["bin"].astype(str))
    )


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * (t["predicted"] - t["observed"]).abs()).sum() / t["n"].sum())


def score_metrics(y, p) -> dict:
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "log_loss": float(log_loss(y, p)),
        "calibration_error": expected_calibration_error(y, p),
    }


def decision_metrics(y, decisions) -> dict:
    """What the approve/refer/decline policy does to a book of customers."""
    y = np.asarray(y)
    d = np.asarray(decisions)
    approved, declined, referred = d == "approve", d == "decline", d == "refer"
    defaulters, payers = y == 1, y == 0
    return {
        "approved_share": float(approved.mean()),
        "referred_share": float(referred.mean()),
        "declined_share": float(declined.mean()),
        "default_rate_if_all_approved": float(y.mean()),
        "default_rate_among_approved": float(y[approved].mean()) if approved.any() else float("nan"),
        "defaulters_not_approved": float((~approved[defaulters]).mean()),
        "defaulters_declined": float(declined[defaulters].mean()),
        "payers_declined": float(declined[payers].mean()),
        "payers_approved": float(approved[payers].mean()),
    }
