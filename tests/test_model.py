import numpy as np

from crediwise import config as cfg
from crediwise import data as data_mod
from crediwise.model import baseline_model, build_search


def _xy(raw_sample):
    df = data_mod.add_features(data_mod.clean(raw_sample))
    return df[cfg.FEATURES], df[cfg.TARGET]


def test_search_returns_a_model_with_probabilities(raw_sample):
    X, y = _xy(raw_sample)
    model = build_search(n_iter=2).fit(X, y).best_estimator_
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_model_finds_the_planted_signal(raw_sample):
    """Late payments drive default in the fixture; the model should notice."""
    from sklearn.metrics import roc_auc_score
    X, y = _xy(raw_sample)
    model = build_search(n_iter=2).fit(X, y).best_estimator_
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) > 0.7


def test_training_is_reproducible(raw_sample):
    X, y = _xy(raw_sample)
    a = build_search(n_iter=2).fit(X, y).best_estimator_.predict_proba(X)
    b = build_search(n_iter=2).fit(X, y).best_estimator_.predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_baseline_trains(raw_sample):
    X, y = _xy(raw_sample)
    assert baseline_model().fit(X, y).predict_proba(X).shape == (len(X), 2)
