"""The decision layer.

Kept separate from the model on purpose. The model estimates probabilities;
the policy decides what to do with them. Business or regulatory changes to the
lending rule should not require retraining, and the cost asymmetry between
error types is a policy choice, not a modelling one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import config as cfg

#: Cost of predicting column j when the truth is row i, in arbitrary units.
#: Approving a genuinely Poor applicant (a default) is set to hurt roughly
#: three times as much as declining a genuinely Good one (a lost customer).
#: These numbers are illustrative -- a real deployment sets them from the
#: lender's own loss and margin data.
DEFAULT_COST_MATRIX = np.array([
    [0.0, 2.0, 5.0],   # true Poor
    [1.0, 0.0, 2.0],   # true Standard
    [1.5, 0.5, 0.0],   # true Good
])


@dataclass
class DecisionPolicy:
    """Turns calibrated probabilities into a lending decision."""

    approve_threshold: float = 0.55
    refer_threshold: float = 0.35
    cost_matrix: np.ndarray = field(default_factory=lambda: DEFAULT_COST_MATRIX)

    def risk_band(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray([cfg.TARGET_LABELS[i] for i in proba.argmax(axis=1)])

    def expected_cost(self, proba: np.ndarray) -> np.ndarray:
        """Expected cost of each possible action given the probabilities."""
        return proba @ self.cost_matrix

    def min_cost_decision(self, proba: np.ndarray) -> np.ndarray:
        """Cost-sensitive choice, which need not match the argmax band."""
        return self.expected_cost(proba).argmin(axis=1)

    def decide(self, proba: np.ndarray) -> list[dict]:
        """Approve / refer / decline, with the probability that drove it."""
        good = proba[:, cfg.TARGET_MAP["Good"]]
        adverse = proba[:, cfg.TARGET_MAP["Poor"]]
        decisions = []
        for i in range(len(proba)):
            if good[i] >= self.approve_threshold:
                outcome = "approve"
            elif good[i] >= self.refer_threshold:
                outcome = "refer"
            else:
                outcome = "decline"
            decisions.append({
                "decision": outcome,
                "is_adverse": outcome == "decline",
                "p_good": round(float(good[i]), 4),
                "p_poor": round(float(adverse[i]), 4),
                "risk_band": cfg.TARGET_LABELS[int(proba[i].argmax())],
                "cost_optimal_band": cfg.TARGET_LABELS[
                    int(self.expected_cost(proba[i:i + 1]).argmin())
                ],
            })
        return decisions

    def to_dict(self) -> dict:
        out = asdict(self)
        out["cost_matrix"] = self.cost_matrix.tolist()
        return out
