"""Scoring service.

    uvicorn crediwise.api.main:app --port 8000

POST /score takes one customer's six-month history and returns the
probability of default next month, a decision (approve / refer / decline)
and, unless approved, the reasons. Requires a trained model: run
`make train` first.
"""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .. import __version__
from .. import config as cfg
from ..data import add_features
from ..explain import Explainer
from .schemas import Applicant, HealthResponse, ScoreResponse

DISCLAIMER = (
    "Demonstration model trained on 2005 credit card data from one Taiwanese "
    "lender. Not validated for real lending decisions."
)

STATE: dict = {"bundle": None, "explainer": None, "model_version": None}


def _file_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest[:12]}"


def load_model() -> None:
    """Load the trained model if present; otherwise run degraded."""
    STATE.update(bundle=None, explainer=None, model_version=None)
    path = cfg.ARTIFACT_PATH
    if not path.exists():
        return
    bundle = joblib.load(path)
    STATE.update(bundle=bundle, explainer=Explainer(bundle["model"]),
                 model_version=_file_version(path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="CrediWise",
    version=__version__,
    description="Predicts whether a credit card customer will miss next "
                "month's payment, with a decision and the reasons behind it.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = STATE["bundle"]
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=STATE["model_version"],
        trained_at=bundle["trained_at"] if bundle else None,
    )


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(status_code=503,
                            detail="No trained model found. Run `make train` first.")

    # The same feature code the model was trained with, so a feature is
    # computed identically here and in training.
    frame = add_features(pd.DataFrame([applicant.model_dump()]))[bundle["features"]]
    p = float(bundle["model"].predict_proba(frame)[0, 1])
    decision = str(bundle["policy"].decide([p])[0])
    reasons = STATE["explainer"].reasons(frame)[0] if decision != "approve" else []

    return ScoreResponse(probability_of_default=round(p, 4), decision=decision,
                         reasons=reasons, model_version=STATE["model_version"],
                         disclaimer=DISCLAIMER)
