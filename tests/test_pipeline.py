import numpy as np
import pandas as pd

from crediwise import data as data_mod
from crediwise import features
from crediwise import config as cfg
from crediwise.model import build_model
from crediwise.splits import holdout_split


def test_preprocessor_emits_no_missing_values(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    X = data_mod.feature_frame(cleaned)
    pre = features.build_preprocessor()
    out = pre.fit_transform(X)
    assert not np.isnan(out).any(), "imputation should leave no NaN"


def test_nominals_are_one_hot_not_ordinal(raw_sample):
    """Audit #7: LabelEncoder imposed a false ordering on nominal categories."""
    cleaned = data_mod.clean(raw_sample)
    pre = features.build_preprocessor().fit(data_mod.feature_frame(cleaned))
    names = features.feature_names(pre)
    occupation_cols = [n for n in names if n.startswith("Occupation_")]
    assert len(occupation_cols) >= 2, "Occupation must expand to indicator columns"


def test_model_trains_and_predicts_probabilities(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    X, y = data_mod.feature_frame(cleaned), cleaned[cfg.TARGET_COL]
    train_idx, test_idx = holdout_split(cleaned)
    model = build_model(n_estimators=20).fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])
    assert proba.shape == (len(test_idx), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_is_deterministic(raw_sample):
    """Audit #9: the notebook seeded nothing, so its numbers were one draw."""
    cleaned = data_mod.clean(raw_sample)
    X, y = data_mod.feature_frame(cleaned), cleaned[cfg.TARGET_COL]
    a = build_model(n_estimators=20).fit(X, y).predict_proba(X)
    b = build_model(n_estimators=20).fit(X, y).predict_proba(X)
    # allclose rather than array_equal: thread-pool aggregation order can shift
    # the last bit of a float even with the forest itself fully seeded.
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
