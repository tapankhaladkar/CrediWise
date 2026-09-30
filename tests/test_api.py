"""API tests.

These build a small model from the synthetic fixture and inject it, so the
suite does not depend on a full `make train` having been run.
"""

import pytest
from fastapi.testclient import TestClient

from crediwise import config as cfg
from crediwise import data as data_mod
from crediwise.api import main as api_main
from crediwise.explain import Explainer
from crediwise.model import build_model
from crediwise.policy import DecisionPolicy
from crediwise.api.schemas import Applicant


@pytest.fixture()
def client(raw_sample, monkeypatch):
    cleaned = data_mod.clean(raw_sample)
    X, y = data_mod.feature_frame(cleaned), cleaned[cfg.TARGET_COL]
    pipeline = build_model(n_estimators=20).fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "calibrated": pipeline,     # uncalibrated stand-in is fine for wiring
        "policy": DecisionPolicy(),
        "feature_columns": list(X.columns),
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setitem(api_main.STATE, "bundle", bundle)
    monkeypatch.setitem(api_main.STATE, "explainer", Explainer(pipeline))
    with TestClient(api_main.app) as c:
        monkeypatch.setitem(api_main.STATE, "bundle", bundle)
        monkeypatch.setitem(api_main.STATE, "explainer", Explainer(pipeline))
        yield c


def _example() -> dict:
    return Applicant.model_config["json_schema_extra"]["examples"][0]


def test_health_reports_model_state(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_score_returns_calibrated_probabilities(client):
    response = client.post("/score", json=_example())
    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == {"Poor", "Standard", "Good"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
    assert body["risk_band"] in {"Poor", "Standard", "Good"}
    assert body["decision"] in {"approve", "refer", "decline"}


def test_partial_record_still_scores(client):
    """Missing fields are imputed, so a sparse application is not rejected."""
    response = client.post("/score", json={"Annual_Income": 25_000})
    assert response.status_code == 200
    assert response.json()["risk_band"] in {"Poor", "Standard", "Good"}


def test_out_of_range_input_is_rejected(client):
    """The contract enforces the same domain ranges as training."""
    payload = _example() | {"Interest_Rate": 5797}
    assert client.post("/score", json=payload).status_code == 422


def test_adverse_outcome_carries_reason_codes(client):
    """ECOA 12 CFR 1002.9: a decline must state its principal reasons."""
    payload = _example() | {
        "Outstanding_Debt": 400_000, "Delay_from_due_date": 180,
        "Num_of_Delayed_Payment": 90, "Credit_Mix": 0,
        "Credit_Utilization_Ratio": 99, "Payment_of_Min_Amount": 1,
        "Num_Credit_Inquiries": 45, "Credit_History_Months": 1,
    }
    body = client.post("/score", json=payload).json()
    if body["is_adverse"]:
        assert body["reason_codes"], "an adverse decision must cite reasons"
        first = body["reason_codes"][0]
        assert first["reason"] and first["reason"] != first["feature"], \
            "reasons should be plain language, not raw column names"


def test_response_always_carries_a_disclaimer(client):
    body = client.post("/score", json=_example()).json()
    assert "not validated for real lending" in body["disclaimer"]
