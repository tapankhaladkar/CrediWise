"""Model construction and probability calibration.

A lending decision needs a probability it can act on, not just an argmax label.
Random Forest votes are not calibrated probabilities, so the deployed estimator
wraps the forest in isotonic regression fit on a customer-disjoint slice of the
training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from . import config as cfg
from .features import build_preprocessor
from .splits import holdout_split


def build_model(include_bureau: bool = True, **rf_kwargs) -> Pipeline:
    """Preprocessing plus a seeded Random Forest, as one fitted object.

    Persisting the whole pipeline is what makes the artifact servable: the
    original notebook saved nothing, and its LabelEncoder mappings were
    unrecoverable once the kernel died (audit #25).
    """
    # min_samples_leaf=20 rather than the sklearn default of 1: it scores
    # marginally better here (67.65 vs 67.41 macro-F1) and keeps the trees
    # shallow enough for SHAP to stay tractable. Fully grown trees average
    # 11,332 leaves, which pushes a 2,000-row explanation from 11 minutes
    # to 94.
    params = dict(
        n_estimators=300,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
        class_weight=None,
    )
    params.update(rf_kwargs)
    return Pipeline([
        ("preprocess", build_preprocessor(include_bureau=include_bureau)),
        ("classifier", RandomForestClassifier(**params)),
    ])


def fit_calibrated(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    calibration_size: float = 0.25,
    method: str = "isotonic",
) -> tuple[CalibratedClassifierCV, dict]:
    """Fit ``model`` then calibrate it on a held-out, customer-disjoint slice.

    ``CalibratedClassifierCV`` cannot be handed groups directly, so the
    fit/calibration split is made here and the base estimator is wrapped in
    ``FrozenEstimator`` so calibration does not refit it. Reusing training
    rows for calibration would produce optimistically flat reliability curves.
    """
    frame = pd.DataFrame({cfg.GROUP_COL: groups.values, cfg.TARGET_COL: y.values})
    fit_idx, cal_idx = holdout_split(frame, test_size=calibration_size)

    base = model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    calibrated.fit(X.iloc[cal_idx], y.iloc[cal_idx])

    meta = {
        "calibration_method": method,
        "fit_rows": int(len(fit_idx)),
        "calibration_rows": int(len(cal_idx)),
        "fit_customers": int(frame[cfg.GROUP_COL].iloc[fit_idx].nunique()),
        "calibration_customers": int(frame[cfg.GROUP_COL].iloc[cal_idx].nunique()),
    }
    return calibrated, meta
