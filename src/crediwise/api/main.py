"""FastAPI scoring service.

    uvicorn crediwise.api.main:app --port 8000
    CREDIWISE_MODEL=nobureau uvicorn crediwise.api.main:app --port 8000

Serves calibrated probabilities, a decision from the policy layer, and reason
codes for adverse outcomes. Requires a trained artifact -- run `make train`
(or `make train-nobureau`) first.

Each deployment serves exactly one model variant, chosen by the
`CREDIWISE_MODEL` environment variable (default `full`). Routing requests
between variants by which fields they carry is deliberately not supported:
applicants without bureau data are disproportionately thin-file, and sending
them all to the weaker no-bureau model would create a two-tier system along
protected lines. See docs/FEATURE_CONTRACT.md.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

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

STATE: dict = {"bundle": None, "explainer": None, "variant": None,
               "model_version": None}


def required_bureau_inputs(bundle: dict) -> list[str]:
    """Bureau features the loaded model was trained on.

    Derived from the model's own feature list rather than hard-coded, so a
    deployment serving the `--no-bureau` variant does not demand them.
    """
    return [c for c in cfg.BUREAU_FEATURES if c in bundle["feature_columns"]]


def selected_variant() -> str:
    """The model variant this deployment is configured to serve.

    An unrecognised value raises rather than falling back to the default: a
    deployment that believes it serves the no-bureau model while actually
    serving the full one is the misconfiguration this setting exists to
    prevent.
    """
    raw = os.environ.get(cfg.MODEL_ENV_VAR, cfg.DEFAULT_MODEL_VARIANT)
    value = raw.strip().lower()
    if value not in cfg.MODEL_VARIANTS:
        raise ValueError(
            f"{cfg.MODEL_ENV_VAR}={raw!r} is not a known model variant. "
            f"Use one of: {', '.join(cfg.MODEL_VARIANTS)}."
        )
    return value


def artifact_path(variant: str) -> Path:
    return cfg.ARTIFACT_DIR / cfg.MODEL_VARIANTS[variant]["artifact"]


def _file_version(path: Path) -> str:
    """Content hash of the artifact: identifies the exact model served."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:12]}"


def load_model() -> None:
    """Load the configured variant.

    A missing artifact leaves the service up but degraded (/health says so,
    /score returns 503). An artifact whose contents contradict its variant --
    say, the full model copied to the no-bureau file name -- fails startup.
    """
    variant = selected_variant()
    STATE.update(bundle=None, explainer=None, variant=variant, model_version=None)

    path = artifact_path(variant)
    if not path.exists():
        return
    bundle = joblib.load(path)

    expects_bureau = cfg.MODEL_VARIANTS[variant]["include_bureau"]
    has_bureau = bool(required_bureau_inputs(bundle))
    if has_bureau != expects_bureau:
        raise RuntimeError(
            f"{path.name} {'uses' if has_bureau else 'does not use'} bureau "
            f"inputs, but {cfg.MODEL_ENV_VAR}={variant} expects a model that "
            f"{'does' if expects_bureau else 'does not'}. The file is probably "
            f"the wrong variant; retrain with "
            f"`make {'train' if expects_bureau else 'train-nobureau'}`."
        )

    STATE.update(
        bundle=bundle,
        explainer=Explainer(bundle["pipeline"]),
        model_version=_file_version(path),
    )


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
        model_variant=STATE.get("variant"),
        model_version=STATE.get("model_version"),
        trained_at=None if not bundle else bundle.get("trained_at"),
        required_bureau_inputs=None if not bundle else required_bureau_inputs(bundle),
    )


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    """Score one applicant-month.

    Most fields may be omitted and are imputed with training medians. Bureau
    inputs the loaded model depends on may not: a request without them is
    rejected with 422 rather than scored on the population median.
    """
    bundle = STATE.get("bundle")
    if bundle is None:
        variant = STATE.get("variant") or cfg.DEFAULT_MODEL_VARIANT
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained model for variant '{variant}': expected "
                f"artifacts/{cfg.MODEL_VARIANTS[variant]['artifact']}. Run "
                f"`make {'train' if variant == 'full' else 'train-nobureau'}` first."
            ),
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
                    "it, or use a deployment serving the no-bureau model "
                    f"({cfg.MODEL_ENV_VAR}=nobureau)."
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
        model_variant=STATE["variant"],
        model_version=STATE["model_version"],
        disclaimer=DISCLAIMER,
    )
