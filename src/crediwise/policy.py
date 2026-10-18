"""Turning a probability into a decision.

Kept separate from the model on purpose: the model estimates the chance of
default, the policy decides what to do about it, and a lender can change its
appetite for risk without retraining.

Approve threshold. Lending to a customer who then defaults is set to cost
five times as much as turning away a customer who would have repaid (lost
balance versus lost profit). Approving is worth it while the expected loss is
smaller than the expected gain forgone:

    p * cost_default  <  (1 - p) * cost_lost_customer
    p  <  cost_lost_customer / (cost_lost_customer + cost_default)  =  1/6

Decline threshold. At 50% the customer is more likely than not to default.
Between the two, the application goes to a person.

The 5:1 ratio is illustrative. A real lender would set it from its own loss
and margin figures; the thresholds follow automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    cost_default: float = 5.0
    cost_lost_customer: float = 1.0
    decline_at: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.approve_below < self.decline_at <= 1:
            raise ValueError(
                f"approve threshold {self.approve_below:.3f} must sit below the "
                f"decline threshold {self.decline_at:.3f}"
            )

    @property
    def approve_below(self) -> float:
        return self.cost_lost_customer / (self.cost_lost_customer + self.cost_default)

    def decide(self, p_default) -> np.ndarray:
        p = np.asarray(p_default, dtype=float)
        return np.where(p < self.approve_below, "approve",
                        np.where(p >= self.decline_at, "decline", "refer"))

    def to_dict(self) -> dict:
        return asdict(self) | {"approve_below": self.approve_below}
