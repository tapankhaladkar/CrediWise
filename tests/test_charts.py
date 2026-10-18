"""Every chart draws, in both themes, from the same inputs the scripts use."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crediwise import charts, evaluate, fairness  # noqa: E402
from crediwise import config as cfg  # noqa: E402
from crediwise import data as data_mod  # noqa: E402
from crediwise.policy import DecisionPolicy  # noqa: E402


@pytest.fixture(scope="module")
def inputs(raw_sample):
    df = data_mod.add_audit_dimensions(data_mod.add_features(data_mod.clean(raw_sample)))
    y = df[cfg.TARGET].to_numpy()
    rng = np.random.default_rng(0)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, len(y)), 0.01, 0.99)
    policy = DecisionPolicy()
    decided = policy.decide(p)
    return {
        "df": df, "y": y, "p": p, "policy": policy, "decided": decided,
        "audit": fairness.audit(y, p, decided, df.reset_index(drop=True), cfg.AUDIT_DIMENSIONS),
        "importance": pd.DataFrame({"feature": cfg.FEATURES,
                                    "mean_abs_shap": np.linspace(1, 0.1, len(cfg.FEATURES))}),
    }


@pytest.mark.parametrize("theme", [charts.LIGHT, charts.DARK], ids=["light", "dark"])
def test_every_chart_draws(inputs, theme):
    figures = [
        charts.default_rate_by_lateness(inputs["df"], theme),
        charts.roc(inputs["y"], inputs["p"], inputs["p"] * 0.9, theme),
        charts.calibration(evaluate.calibration_table(inputs["y"], inputs["p"]), theme),
        charts.decisions(inputs["y"], inputs["decided"], inputs["policy"], theme),
        charts.importance(inputs["importance"], theme),
        charts.fairness(inputs["audit"], theme),
    ]
    for fig in figures:
        assert fig.get_facecolor()[:3] == matplotlib.colors.to_rgb(theme.surface)
        plt.close(fig)


def test_dark_and_light_differ():
    assert charts.LIGHT.surface != charts.DARK.surface
    assert charts.LIGHT.blue != charts.DARK.blue
