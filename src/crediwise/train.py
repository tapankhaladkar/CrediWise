"""Train, evaluate once on held-out data, and save the model.

    python -m crediwise.train

1. Split off 20% of customers as a test set and do not look at it again.
2. On the other 80%: benchmark logistic regression, then search
   gradient-boosting settings by 5-fold cross-validation.
3. Score the chosen model on the test set -- once.
4. Apply the decision policy, audit fairness, compute explanations.
5. Save the model to artifacts/ and the results to reports/.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from . import __version__
from . import config as cfg
from . import data as data_mod
from . import evaluate, fairness
from .explain import Explainer
from .model import baseline_model, build_search, cv_splitter
from .policy import DecisionPolicy


def log(msg: str) -> None:
    print(msg, flush=True)


def _json(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not serialisable: {type(obj)}")


def run() -> dict:
    started = time.time()
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading data...")
    raw = data_mod.load_raw()
    df = data_mod.add_audit_dimensions(data_mod.add_features(data_mod.clean(raw)))
    log(f"  {len(raw):,} rows, {len(raw) - len(df)} duplicates dropped, "
        f"{len(df):,} customers, default rate {df[cfg.TARGET].mean():.1%}")

    train, test = train_test_split(df, test_size=cfg.TEST_SIZE, stratify=df[cfg.TARGET],
                                   random_state=cfg.RANDOM_STATE)
    X_train, y_train = train[cfg.FEATURES], train[cfg.TARGET]
    X_test, y_test = test[cfg.FEATURES], test[cfg.TARGET]
    log(f"  train {len(train):,} / test {len(test):,} (test set held back)")

    log("Benchmark: logistic regression, 5-fold CV on training data...")
    lr_cv = cross_val_score(baseline_model(), X_train, y_train, cv=cv_splitter(),
                            scoring="roc_auc", n_jobs=-1)
    log(f"  CV AUC {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}")

    log(f"Searching gradient-boosting settings ({cfg.SEARCH_ITERATIONS} candidates x "
        f"{cfg.N_SPLITS} folds)...")
    search = build_search().fit(X_train, y_train)
    model = search.best_estimator_
    best_idx = search.best_index_
    cv_std = float(search.cv_results_["std_test_score"][best_idx])
    log(f"  best CV AUC {search.best_score_:.4f} +/- {cv_std:.4f}")

    log("Scoring the held-out test set (first and only time)...")
    p_test = model.predict_proba(X_test)[:, 1]
    lr_test_auc = float(roc_auc_score(
        y_test, baseline_model().fit(X_train, y_train).predict_proba(X_test)[:, 1]))
    scores = evaluate.score_metrics(y_test, p_test)
    log(f"  AUC {scores['auc']:.4f} | calibration error {scores['calibration_error']:.4f}")

    policy = DecisionPolicy()
    decisions = policy.decide(p_test)
    outcomes = evaluate.decision_metrics(y_test, decisions)

    log("Auditing fairness...")
    audit = fairness.audit(y_test, p_test, decisions, test.reset_index(drop=True),
                           cfg.AUDIT_DIMENSIONS)

    log("Explaining...")
    importance = Explainer(model).global_importance(X_test)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    params = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
              for k, v in search.best_params_.items()}
    joblib.dump({"model": model, "policy": policy, "features": cfg.FEATURES,
                 "trained_at": trained_at, "params": params, "version": __version__},
                cfg.ARTIFACT_PATH, compress=3)
    log(f"  saved {cfg.ARTIFACT_PATH.name} ({cfg.ARTIFACT_PATH.stat().st_size / 1e6:.1f} MB)")

    report = {
        "trained_at": trained_at,
        "data": {"rows": int(len(raw)), "duplicates_dropped": int(len(raw) - len(df)),
                 "customers": int(len(df)), "train": int(len(train)), "test": int(len(test)),
                 "default_rate": float(df[cfg.TARGET].mean())},
        "selection": {
            "logistic_regression_cv_auc": float(lr_cv.mean()),
            "logistic_regression_cv_std": float(lr_cv.std()),
            "gradient_boosting_cv_auc": float(search.best_score_),
            "gradient_boosting_cv_std": cv_std,
            "search_candidates": cfg.SEARCH_ITERATIONS,
            "best_params": params,
            "trees_after_early_stopping": int(model.n_iter_),
        },
        "test": scores | {"logistic_regression_auc": lr_test_auc},
        "calibration": evaluate.calibration_table(y_test, p_test).to_dict(orient="records"),
        "policy": policy.to_dict(),
        "decisions": outcomes,
        "fairness": audit,
        "importance": importance.to_dict(orient="records"),
        "features": cfg.FEATURES,
        "train_seconds": round(time.time() - started, 1),
    }
    (cfg.REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=_json))
    log(f"  saved metrics.json  [{report['train_seconds']}s total]")
    return report


def main() -> None:
    r = run()
    t, d = r["test"], r["decisions"]
    print("\n" + "=" * 60)
    print(f"  AUC              {t['auc']:.4f}   (logistic regression {t['logistic_regression_auc']:.4f})")
    print(f"  KS               {t['ks']:.4f}")
    print(f"  PR-AUC           {t['pr_auc']:.4f}   (random: {t['default_rate']:.4f})")
    print(f"  Brier            {t['brier']:.4f}   (no-skill: {t['brier_baseline']:.4f})")
    print(f"  calibration err  {t['calibration_error']:.4f}")
    print(f"  approve {d['approved_share']:.1%} | refer {d['referred_share']:.1%} | "
          f"decline {d['declined_share']:.1%}")
    print(f"  default rate among approved {d['default_rate_among_approved']:.1%} "
          f"(vs {d['default_rate_if_all_approved']:.1%} if everyone were approved)")
    print("=" * 60)


if __name__ == "__main__":
    main()
