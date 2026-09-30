"""Training entry point.

Produces every number the README quotes, plus the artifacts the API serves:

    python -m crediwise.train                # full model
    python -m crediwise.train --no-bureau    # variant without Credit_Mix
    python -m crediwise.train --demo-leakage # also report the leaky split

Outputs go to artifacts/ (the servable model) and reports/ (metrics, fairness).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from . import config as cfg
from . import data as data_mod
from . import evaluate as eval_mod
from . import fairness as fairness_mod
from .explain import Explainer
from .model import build_model, fit_calibrated
from .policy import DecisionPolicy
from .splits import assert_no_group_overlap, cv_splitter, holdout_split


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")


def run(include_bureau: bool = True, demo_leakage: bool = False) -> dict:
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning...")
    raw = data_mod.load_raw()
    df = data_mod.add_audit_dimensions(data_mod.clean(raw))
    X = data_mod.feature_frame(df, include_bureau=include_bureau)
    y = df[cfg.TARGET_COL]
    groups = df[cfg.GROUP_COL]
    print(f"  {len(df):,} rows, {groups.nunique():,} customers, "
          f"{X.shape[1]} raw features")

    train_idx, test_idx = holdout_split(df)
    assert_no_group_overlap(groups, train_idx, test_idx)
    print(f"  grouped holdout: {len(train_idx):,} train / {len(test_idx):,} test, "
          f"0 shared customers")

    print("Cross-validating (StratifiedGroupKFold, 5 folds)...")
    cv_scores = cross_val_score(
        build_model(include_bureau=include_bureau),
        X.iloc[train_idx], y.iloc[train_idx],
        groups=groups.iloc[train_idx],
        cv=cv_splitter(), scoring="f1_macro", n_jobs=1,
    )
    print(f"  macro-F1 {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}")

    print("Fitting and calibrating...")
    uncalibrated = build_model(include_bureau=include_bureau).fit(
        X.iloc[train_idx], y.iloc[train_idx]
    )
    calibrated, calib_meta = fit_calibrated(
        build_model(include_bureau=include_bureau),
        X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx],
    )

    y_test = y.iloc[test_idx].to_numpy()
    proba_uncal = uncalibrated.predict_proba(X.iloc[test_idx])
    proba_cal = calibrated.predict_proba(X.iloc[test_idx])

    metrics = {
        "uncalibrated": eval_mod.evaluate(y_test, proba_uncal, "uncalibrated"),
        "calibrated": eval_mod.evaluate(y_test, proba_cal, "calibrated"),
        "cross_validation": {
            "metric": "f1_macro",
            "folds": cfg.N_SPLITS,
            "scores": cv_scores.tolist(),
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        },
        "calibration_setup": calib_meta,
    }

    if demo_leakage:
        print("Reproducing the leaky split for comparison...")
        tr_l, te_l = train_test_split(
            np.arange(len(df)), test_size=cfg.TEST_SIZE,
            random_state=cfg.RANDOM_STATE,
        )
        leaky = build_model(include_bureau=include_bureau).fit(
            X.iloc[tr_l], y.iloc[tr_l]
        )
        leaky_proba = leaky.predict_proba(X.iloc[te_l])
        shared = len(set(groups.iloc[tr_l]) & set(groups.iloc[te_l]))
        metrics["leaky_split_demo"] = eval_mod.evaluate(
            y.iloc[te_l].to_numpy(), leaky_proba, "random row split (LEAKY)"
        )
        metrics["leaky_split_demo"]["shared_customers"] = shared
        print(f"  leaky accuracy {metrics['leaky_split_demo']['accuracy']*100:.2f}% "
              f"vs honest {metrics['calibrated']['accuracy']*100:.2f}% "
              f"({shared:,} customers shared)")

    print("Auditing fairness...")
    audit_frame = df.iloc[test_idx][cfg.AUDIT_DIMENSIONS].reset_index(drop=True)
    fairness_report = fairness_mod.audit(y_test, proba_cal, audit_frame)

    print("Computing SHAP importances...")
    sample = X.iloc[test_idx].sample(
        min(1000, len(test_idx)), random_state=cfg.RANDOM_STATE
    )
    explainer = Explainer(uncalibrated)
    importance = explainer.global_importance(sample)

    policy = DecisionPolicy()
    decisions = policy.decide(proba_cal[:5])

    bundle = {
        "pipeline": uncalibrated,
        "calibrated": calibrated,
        "policy": policy,
        "include_bureau": include_bureau,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(X.columns),
        "cv_macro_f1": float(cv_scores.mean()),
        "test_macro_f1": metrics["calibrated"]["macro_f1"],
    }
    suffix = "" if include_bureau else "_nobureau"
    model_path = cfg.ARTIFACT_DIR / f"crediwise_model{suffix}.joblib"
    joblib.dump(bundle, model_path, compress=3)
    print(f"  saved {model_path.name} "
          f"({model_path.stat().st_size / 1e6:.1f} MB)")

    report = {
        "trained_at": bundle["trained_at"],
        "include_bureau": include_bureau,
        "n_rows": int(len(df)),
        "n_customers": int(groups.nunique()),
        "metrics": metrics,
        "fairness": fairness_report,
        "global_importance": importance.head(15).to_dict(orient="records"),
        "policy": policy.to_dict(),
        "example_decisions": decisions,
    }
    report_path = cfg.REPORT_DIR / f"metrics{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_safe))
    eval_mod.reliability_table(y_test, proba_cal).to_csv(
        cfg.REPORT_DIR / f"reliability{suffix}.csv", index=False
    )
    print(f"  saved {report_path.name}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CrediWise model.")
    parser.add_argument("--no-bureau", action="store_true",
                        help="train without the bureau-supplied Credit_Mix feature")
    parser.add_argument("--demo-leakage", action="store_true",
                        help="also report the original leaky random split")
    args = parser.parse_args()
    report = run(include_bureau=not args.no_bureau, demo_leakage=args.demo_leakage)

    m = report["metrics"]["calibrated"]
    print("\n" + "=" * 62)
    print(f"  accuracy        {m['accuracy']*100:6.2f}%   "
          f"(baseline {m['baseline_accuracy']*100:.2f}%)")
    print(f"  macro-F1        {m['macro_f1']*100:6.2f}%")
    print(f"  macro AUC (OvR) {m['macro_auc_ovr']:6.3f}")
    print(f"  ECE             {m['expected_calibration_error']:6.4f}")
    print("=" * 62)
    print(m["report"])


if __name__ == "__main__":
    main()
