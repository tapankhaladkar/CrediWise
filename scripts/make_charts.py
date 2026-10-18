"""Render the README charts into docs/images/, light and dark.

    python scripts/make_charts.py      (run by `make train`)

Draws from the trained model and reports/metrics.json, and stops if the two
disagree, so the images can never mix results from different training runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from crediwise import charts  # noqa: E402
from crediwise import config as cfg  # noqa: E402
from crediwise import data  # noqa: E402
from crediwise.model import baseline_model  # noqa: E402

OUT = ROOT / "docs" / "images"


def main() -> None:
    metrics = cfg.REPORT_DIR / "metrics.json"
    if not (metrics.exists() and cfg.ARTIFACT_PATH.exists()):
        sys.exit("No trained model or report; run `make train` first.")
    report = json.loads(metrics.read_text())
    bundle = joblib.load(cfg.ARTIFACT_PATH)

    df = data.load_dataset()
    train, test = data.split(df)
    y = test[cfg.TARGET]
    p = bundle["model"].predict_proba(test[cfg.FEATURES])[:, 1]
    if abs(roc_auc_score(y, p) - report["test"]["auc"]) > 1e-9:
        sys.exit("The saved model and reports/metrics.json disagree; run `make train`.")
    p_benchmark = (baseline_model().fit(train[cfg.FEATURES], train[cfg.TARGET])
                   .predict_proba(test[cfg.FEATURES])[:, 1])

    figures = {
        "default-rate-by-lateness": lambda t: charts.default_rate_by_lateness(df, t),
        "roc": lambda t: charts.roc(y, p, p_benchmark, t),
        "calibration": lambda t: charts.calibration(pd.DataFrame(report["calibration"]), t),
        "decisions": lambda t: charts.decisions(y, bundle["policy"].decide(p),
                                                bundle["policy"], t),
        "importance": lambda t: charts.importance(pd.DataFrame(report["importance"]), t),
        "fairness": lambda t: charts.fairness(report["fairness"], t),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in figures.items():
        for theme in (charts.LIGHT, charts.DARK):
            fig = draw(theme)
            fig.savefig(OUT / f"{name}-{theme.name}.png", dpi=144, bbox_inches="tight",
                        facecolor=theme.surface, metadata={"Software": None})
            plt.close(fig)
    print(f"Wrote {len(figures) * 2} charts to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
