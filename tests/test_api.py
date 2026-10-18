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


def _serve(raw_sample, monkeypatch, include_bureau: bool):
    cleaned = data_mod.clean(raw_sample)
    X = data_mod.feature_frame(cleaned, include_bureau=include_bureau)
    y = cleaned[cfg.TARGET_COL]
    pipeline = build_model(include_bureau=include_bureau, n_estimators=20).fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "calibrated": pipeline,     # uncalibrated stand-in is fine for wiring
        "policy": DecisionPolicy(),
        "include_bureau": include_bureau,
        "feature_columns": list(X.columns),
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    # Set inside the client context too: app startup loads any real artifact
    # from artifacts/, which would otherwise replace the injected model.
    monkeypatch.setitem(api_main.STATE, "bundle", bundle)
    monkeypatch.setitem(api_main.STATE, "explainer", Explainer(pipeline))
    client = TestClient(api_main.app)
    client.__enter__()
    monkeypatch.setitem(api_main.STATE, "bundle", bundle)
    monkeypatch.setitem(api_main.STATE, "explainer", Explainer(pipeline))
    return client


@pytest.fixture()
def client(raw_sample, monkeypatch):
    """Serves a model trained with the bureau feature, like the default build."""
    c = _serve(raw_sample, monkeypatch, include_bureau=True)
    yield c
    c.__exit__(None, None, None)


@pytest.fixture()
def nobureau_client(raw_sample, monkeypatch):
    """Serves a model trained with --no-bureau."""
    c = _serve(raw_sample, monkeypatch, include_bureau=False)
    yield c
    c.__exit__(None, None, None)


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
    """Missing applicant fields are imputed, so a sparse application scores.

    The bureau field is supplied: it is the one input that is never imputed.
    """
    response = client.post("/score", json={"Annual_Income": 25_000, "Credit_Mix": 1})
    assert response.status_code == 200
    assert response.json()["risk_band"] in {"Poor", "Standard", "Good"}


def test_missing_bureau_input_is_rejected_not_imputed(client):
    """docs/FEATURE_CONTRACT.md: never serve the full model with Credit_Mix
    imputed to the median. Without it, Good-class recall drops 0.652 -> 0.491.
    """
    payload = {k: v for k, v in _example().items() if k != "Credit_Mix"}
    response = client.post("/score", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "missing_bureau_input"
    assert detail["missing"] == ["Credit_Mix"]
    assert "--no-bureau" in detail["message"]


def test_explicit_null_bureau_input_is_rejected(client):
    """Sending null is the same as omitting the field."""
    response = client.post("/score", json=_example() | {"Credit_Mix": None})
    assert response.status_code == 422
    assert response.json()["detail"]["missing"] == ["Credit_Mix"]


def test_nobureau_model_does_not_require_credit_mix(nobureau_client):
    """The rule follows the served model's features, not a hard-coded field,
    so a --no-bureau deployment scores requests that lack it.
    """
    payload = {k: v for k, v in _example().items() if k != "Credit_Mix"}
    response = nobureau_client.post("/score", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_band"] in {"Poor", "Standard", "Good"}


def test_nobureau_model_ignores_a_supplied_credit_mix(nobureau_client):
    """A caller that sends the field anyway is not penalised for it."""
    response = nobureau_client.post("/score", json=_example())
    assert response.status_code == 200


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
