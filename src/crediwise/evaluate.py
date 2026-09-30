"""Metrics.

Accuracy alone is misleading on a 53/29/18 class split -- always predicting
"Standard" scores 52.9% (audit #12, #13). Everything here is reported per class
and against that baseline, plus the discrimination and calibration measures a
credit model is normally judged on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config as cfg


def ks_statistic(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between positives and negatives."""
    order = np.argsort(scores)
    y = np.asarray(y_true_binary)[order]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    cum_pos = np.cumsum(y) / pos
    cum_neg = np.cumsum(1 - y) / neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins."""
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def evaluate(y_true, proba, label: str = "model") -> dict:
    """Full metric set for a multiclass credit model."""
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(axis=1)
    classes = sorted(cfg.TARGET_LABELS)

    majority = np.bincount(y_true, minlength=len(classes)).argmax()
    baseline_acc = float((y_true == majority).mean())

    per_class = {}
    for k in classes:
        binary = (y_true == k).astype(int)
        name = cfg.TARGET_LABELS[k]
        auc = roc_auc_score(binary, proba[:, k])
        per_class[name] = {
            "auc": float(auc),
            "gini": float(2 * auc - 1),
            "ks": ks_statistic(binary, proba[:, k]),
            "brier": float(brier_score_loss(binary, proba[:, k])),
        }

    return {
        "label": label,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "baseline_accuracy": baseline_acc,
        "lift_over_baseline": float(accuracy_score(y_true, y_pred) - baseline_acc),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr",
                                             average="macro")),
        "expected_calibration_error": expected_calibration_error(y_true, proba),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred,
            target_names=[cfg.TARGET_LABELS[k] for k in classes],
            digits=3, zero_division=0,
        ),
    }


def reliability_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Confidence vs. observed accuracy, the numbers behind a reliability plot."""
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true)).astype(float)
    bins = pd.cut(confidence, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return (
        pd.DataFrame({"confidence": confidence, "correct": correct, "bin": bins})
        .groupby("bin", observed=True)
        .agg(n=("correct", "size"),
             mean_confidence=("confidence", "mean"),
             observed_accuracy=("correct", "mean"))
        .reset_index()
    )
