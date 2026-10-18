"""API tests.

These build a small model from the synthetic fixture and inject it, so the
suite does not depend on a full `make train` having been run.
"""

import joblib
import pytest
from fastapi.testclient import TestClient

from crediwise import config as cfg
from crediwise import data as data_mod
from crediwise.api import main as api_main
from crediwise.model import build_model
from crediwise.policy import DecisionPolicy
from crediwise.api.schemas import Applicant


def _bundle(raw_sample, include_bureau: bool) -> dict:
    cleaned = data_mod.clean(raw_sample)
    X = data_mod.feature_frame(cleaned, include_bureau=include_bureau)
    y = cleaned[cfg.TARGET_COL]
    pipeline = build_model(include_bureau=include_bureau, n_estimators=20).fit(X, y)
    return {
        "pipeline": pipeline,
        "calibrated": pipeline,     # uncalibrated stand-in is fine for wiring
        "policy": DecisionPolicy(),
        "variant": cfg.variant_for(include_bureau),
        "include_bureau": include_bureau,
        "feature_columns": list(X.columns),
        "trained_at": "2026-01-01T00:00:00+00:00",
    }


def _deploy(tmp_path, monkeypatch, variant: str, bundle: dict | None = None,
            artifact_variant: str | None = None):
    """Point the app at a scratch artifact directory and select a variant.

    The app then loads the model through its real startup path. Using a
    scratch directory also keeps the tests independent of whatever happens to
    be in artifacts/ on this machine.
    """
    monkeypatch.setattr(cfg, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setenv(cfg.MODEL_ENV_VAR, variant)
    if bundle is not None:
        name = cfg.MODEL_VARIANTS[artifact_variant or variant]["artifact"]
        joblib.dump(bundle, tmp_path / name)


@pytest.fixture()
def client(raw_sample, tmp_path, monkeypatch):
    """Serves the full model, as the default deployment does."""
    _deploy(tmp_path, monkeypatch, "full", _bundle(raw_sample, include_bureau=True))
    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture()
def nobureau_client(raw_sample, tmp_path, monkeypatch):
    """Serves the model trained with --no-bureau (CREDIWISE_MODEL=nobureau)."""
    _deploy(tmp_path, monkeypatch, "nobureau", _bundle(raw_sample, include_bureau=False))
    with TestClient(api_main.app) as c:
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
    assert "CREDIWISE_MODEL=nobureau" in detail["message"], \
        "the error should name the deployable alternative"


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


# --- Model selection (CREDIWISE_MODEL) ---------------------------------------

def test_default_deployment_serves_full_model(monkeypatch):
    monkeypatch.delenv(cfg.MODEL_ENV_VAR, raising=False)
    assert api_main.selected_variant() == "full"


def test_variant_name_tolerates_case_and_whitespace(monkeypatch):
    monkeypatch.setenv(cfg.MODEL_ENV_VAR, "  NoBureau ")
    assert api_main.selected_variant() == "nobureau"


def test_unknown_variant_fails_startup_instead_of_falling_back(tmp_path, monkeypatch):
    """A typo must not silently serve the full model."""
    _deploy(tmp_path, monkeypatch, "no-bureau")
    with pytest.raises(ValueError, match="full, nobureau"):
        with TestClient(api_main.app):
            pass


def test_health_reports_what_is_served(client):
    body = client.get("/health").json()
    assert body["model_variant"] == "full"
    assert body["model_version"].startswith("sha256:")
    assert body["required_bureau_inputs"] == ["Credit_Mix"]


def test_nobureau_health_requires_nothing(nobureau_client):
    body = nobureau_client.get("/health").json()
    assert body["model_variant"] == "nobureau"
    assert body["required_bureau_inputs"] == []


def test_every_decision_records_which_model_made_it(raw_sample, tmp_path, monkeypatch):
    # One process serves one variant, so deploy them one after the other.
    responses = {}
    for variant, include_bureau in (("full", True), ("nobureau", False)):
        directory = tmp_path / variant
        directory.mkdir()
        _deploy(directory, monkeypatch, variant, _bundle(raw_sample, include_bureau))
        with TestClient(api_main.app) as c:
            responses[variant] = c.post("/score", json=_example()).json()

    assert responses["full"]["model_variant"] == "full"
    assert responses["nobureau"]["model_variant"] == "nobureau"
    assert responses["full"]["model_version"].startswith("sha256:")
    assert responses["full"]["model_version"] != responses["nobureau"]["model_version"], \
        "different model files must report different versions"


def test_missing_artifact_degrades_and_says_what_to_build(tmp_path, monkeypatch):
    _deploy(tmp_path, monkeypatch, "nobureau")      # no artifact written
    with TestClient(api_main.app) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_variant"] == "nobureau"
        response = c.post("/score", json=_example())
        assert response.status_code == 503
        assert "crediwise_model_nobureau.joblib" in response.json()["detail"]
        assert "make train-nobureau" in response.json()["detail"]


def test_wrong_model_under_a_variant_name_fails_startup(raw_sample, tmp_path, monkeypatch):
    """The full model copied to the no-bureau file name must not be served
    as if it were the no-bureau model -- it would reject every request
    lacking Credit_Mix while claiming not to need it.
    """
    _deploy(tmp_path, monkeypatch, "nobureau",
            bundle=_bundle(raw_sample, include_bureau=True))
    with pytest.raises(RuntimeError, match="wrong variant"):
        with TestClient(api_main.app):
            pass

