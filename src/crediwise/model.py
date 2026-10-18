"""The model.

Gradient boosting on 24 numeric credit-behaviour features. There is no
preprocessing pipeline because none is needed: the data has no missing values
and no categorical inputs, and tree models do not need scaling.

No probability calibration layer either. Measured on out-of-fold training
predictions, the raw model is already well calibrated (expected calibration
error 0.0063); sigmoid calibration changed nothing and isotonic made Brier
score and AUC slightly worse. A step that does not help is a step that can
break, so it was left out.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_STATE)


def base_model(**params) -> HistGradientBoostingClassifier:
    """Gradient boosting with early stopping on an internal validation slice."""
    return HistGradientBoostingClassifier(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=cfg.RANDOM_STATE,
        **params,
    )


def build_search(n_iter: int = cfg.SEARCH_ITERATIONS) -> RandomizedSearchCV:
    """Randomised hyperparameter search, scored by cross-validated AUC.

    Run on the training split only; the test set is not seen until the
    single final evaluation.
    """
    space = {
        "learning_rate": loguniform(0.02, 0.3),
        "max_leaf_nodes": randint(8, 64),
        "min_samples_leaf": randint(20, 300),
        "l2_regularization": loguniform(1e-3, 10),
        "max_features": uniform(0.4, 0.6),
    }
    return RandomizedSearchCV(
        base_model(), space, n_iter=n_iter, cv=cv_splitter(),
        scoring="roc_auc", n_jobs=-1, random_state=cfg.RANDOM_STATE,
    )


def baseline_model() -> Pipeline:
    """Logistic regression, the simple benchmark the boosted model must beat."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
