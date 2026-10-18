"""Service tests.

Each test trains a small model on the fixture, saves it to a scratch folder
and lets the app load it through its normal startup path, so the suite does
not depend on `make train` or on whatever model sits in artifacts/.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crediwise import config as cfg
from crediwise import data as data_mod
from crediwise.api import main as api_main
from crediwise.api.schemas import Applicant
from crediwise.model import build_search
from crediwise.policy import DecisionPolicy

EXAMPLE = Applicant.model_config["json_schema_extra"]["examples"][0]

LOW_RISK = {"credit_limit": 300_000,
            **{s: -1 for s in cfg.STATUS},
            **{b: 2_000 for b in cfg.BILLS},
            **{p: 2_000 for p in cfg.PAID}}
HIGH_RISK = {"credit_limit": 20_000,
             **{s: 3 for s in cfg.STATUS},
             **{b: 19_500 for b in cfg.BILLS},
             **{p: 0 for p in cfg.PAID}}


@pytest.fixture(scope="module")
def trained(raw_sample):
    df = data_mod.add_features(data_mod.clean(raw_sample))
    return build_search(n_iter=2).fit(df[cfg.FEATURES], df[cfg.TARGET]).best_estimator_


@pytest.fixture()
def client(trained, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": trained, "policy": DecisionPolicy(), "features": cfg.FEATURES,
                 "trained_at": "2026-01-01T00:00:00+00:00", "params": {}, "version": "test"},
                path)
    monkeypatch.setattr(cfg, "ARTIFACT_PATH", path)
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert body["model_version"].startswith("sha256:")


def test_score_shape(client):
    body = client.post("/score", json=EXAMPLE).json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "refer", "decline"}
    assert body["model_version"].startswith("sha256:")
    assert "Not validated for real lending" in body["disclaimer"]


def test_risky_history_scores_higher_and_explains_itself(client):
    low = client.post("/score", json=LOW_RISK).json()
    high = client.post("/score", json=HIGH_RISK).json()
    assert high["probability_of_default"] > low["probability_of_default"]
    assert high["decision"] != "approve"
    assert high["reasons"], "a referral or decline must say why"
    assert high["reasons"][0]["reason"] != high["reasons"][0]["feature"], \
        "reasons should be plain language, not column names"


def test_approvals_carry_no_reasons(client):
    body = client.post("/score", json=LOW_RISK).json()
    if body["decision"] == "approve":
        assert body["reasons"] == []


def test_service_matches_training_code(client, trained):
    """The service computes features with the training code, so the same
    customer gets the same probability either way."""
    expected = trained.predict_proba(
        data_mod.add_features(pd.DataFrame([EXAMPLE]))[cfg.FEATURES])[0, 1]
    served = client.post("/score", json=EXAMPLE).json()["probability_of_default"]
    assert served == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("change", [
    {"status_sep": 10},          # status codes stop at 9
    {"paid_aug": -1},            # payments cannot be negative
    {"credit_limit": 0},         # a limit must be positive
])
def test_out_of_range_input_is_rejected(client, change):
    assert client.post("/score", json=EXAMPLE | change).status_code == 422


def test_missing_field_is_rejected(client):
    """The lender's own records hold every field, so nothing is guessed."""
    payload = {k: v for k, v in EXAMPLE.items() if k != "bill_jun"}
    assert client.post("/score", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["sex", "age", "marital_status", "education"])
def test_personal_attributes_are_refused(client, field):
    """Sending them is an error, so no client can think they are used."""
    response = client.post("/score", json=EXAMPLE | {field: 1})
    assert response.status_code == 422


def test_without_a_model_the_service_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "ARTIFACT_PATH", tmp_path / "missing.joblib")
    with TestClient(api_main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        response = c.post("/score", json=EXAMPLE)
        assert response.status_code == 503
        assert "make train" in response.json()["detail"]
