"""FastAPI scoring service.

    uvicorn crediwise.api.main:app --port 8000

Serves calibrated probabilities, a decision from the policy layer, and reason
codes for adverse outcomes. Requires a trained artifact -- run `make train`
first.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .. import config as cfg
from ..explain import Explainer
from .schemas import Applicant, HealthResponse, ScoreResponse

DISCLAIMER = (
    "Model output for evaluation purposes. Trained on a public synthetic "
    "dataset; not validated for real lending decisions and not a substitute "
    "for a compliant adverse-action process."
)

STATE: dict = {"bundle": None, "explainer": None}


def _artifact_path():
    return cfg.ARTIFACT_DIR / "crediwise_model.joblib"


def load_model() -> None:
    path = _artifact_path()
    if not path.exists():
        return
    bundle = joblib.load(path)
    STATE["bundle"] = bundle
    STATE["explainer"] = Explainer(bundle["pipeline"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    STATE.clear()


app = FastAPI(
    title="CrediWise",
    version="0.2.0",
    description="Credit risk banding with calibrated probabilities, "
                "cost-sensitive decisions and ECOA-style reason codes.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = STATE.get("bundle")
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=None if not bundle else "0.2.0",
        trained_at=None if not bundle else bundle.get("trained_at"),
    )


def required_bureau_inputs(bundle: dict) -> list[str]:
    """Bureau features the loaded model was trained on.

    Derived from the model's own feature list rather than hard-coded, so a
    deployment serving the `--no-bureau` variant does not demand them.
    """
    return [c for c in cfg.BUREAU_FEATURES if c in bundle["feature_columns"]]


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    """Score one applicant-month.

    Most fields may be omitted and are imputed with training medians. Bureau
    inputs the loaded model depends on may not: a request without them is
    rejected with 422 rather than scored on the population median.
    """
    bundle = STATE.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run `make train` first.",
        )

    # docs/FEATURE_CONTRACT.md: never serve the full model with Credit_Mix
    # imputed. It is the model's bureau credit-quality input, and filling it
    # with the median substitutes the population average for the field whose
    # absence costs a quarter of Good-class recall (0.652 -> 0.491).
    missing = [c for c in required_bureau_inputs(bundle)
               if getattr(applicant, c) is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_bureau_input",
                "missing": missing,
                "message": (
                    f"The served model requires {', '.join(missing)}, a "
                    "bureau-supplied input, and will not impute it. Supply "
                    "it, or score against a model trained with --no-bureau."
                ),
            },
        )

    frame = pd.DataFrame([applicant.model_dump()])
    frame = frame[bundle["feature_columns"]]

    proba = bundle["calibrated"].predict_proba(frame)
    decision = bundle["policy"].decide(proba)[0]

    reasons = []
    if decision["is_adverse"]:
        reasons = STATE["explainer"].reason_codes(frame, top_n=4)[0]

    return ScoreResponse(
        risk_band=decision["risk_band"],
        probabilities={
            cfg.TARGET_LABELS[i]: round(float(proba[0, i]), 4)
            for i in sorted(cfg.TARGET_LABELS)
        },
        decision=decision["decision"],
        is_adverse=decision["is_adverse"],
        cost_optimal_band=decision["cost_optimal_band"],
        reason_codes=reasons,
        model_version="0.2.0",
        disclaimer=DISCLAIMER,
    )
