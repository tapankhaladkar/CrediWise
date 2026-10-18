"""Request and response shapes for the scoring service.

The request is what a card issuer already holds for an existing customer:
the credit limit it granted and the last six months of statements, payments
and repayment status. Nothing about the customer as a person -- no sex,
marital status, age or education -- is asked for, because the model does not
use it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

STATUS_HELP = ("Repayment status: -2 no balance, -1 paid in full, 0 revolving "
               "credit (paid at least the minimum), 1-9 months late. -2 and 0 "
               "are undocumented in the source and interpreted from the data.")

Status = Field(ge=-2, le=9, description=STATUS_HELP)
Bill = Field(ge=-1_000_000, le=10_000_000,
             description="Statement balance; negative means the customer is in credit")
Paid = Field(ge=0, le=10_000_000, description="Amount paid during the month")


class Applicant(BaseModel):
    credit_limit: float = Field(gt=0, le=10_000_000, description="Credit limit granted")

    status_sep: int = Status
    status_aug: int = Status
    status_jul: int = Status
    status_jun: int = Status
    status_may: int = Status
    status_apr: int = Status

    bill_sep: float = Bill
    bill_aug: float = Bill
    bill_jul: float = Bill
    bill_jun: float = Bill
    bill_may: float = Bill
    bill_apr: float = Bill

    paid_sep: float = Paid
    paid_aug: float = Paid
    paid_jul: float = Paid
    paid_jun: float = Paid
    paid_may: float = Paid
    paid_apr: float = Paid

    # Unknown fields are rejected, not ignored: a client sending sex or age
    # should learn immediately that the model does not use them.
    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [{
        "credit_limit": 120000,
        "status_sep": -1, "status_aug": 2, "status_jul": 0,
        "status_jun": 0, "status_may": 0, "status_apr": 2,
        "bill_sep": 2682, "bill_aug": 1725, "bill_jul": 2682,
        "bill_jun": 3272, "bill_may": 3455, "bill_apr": 3261,
        "paid_sep": 0, "paid_aug": 1000, "paid_jul": 1000,
        "paid_jun": 1000, "paid_may": 0, "paid_apr": 2000,
    }]}}


class Reason(BaseModel):
    feature: str
    reason: str
    contribution: float = Field(description="Push toward default, in log-odds")


class ScoreResponse(BaseModel):
    probability_of_default: float = Field(
        description="Chance the customer misses next month's payment")
    decision: Literal["approve", "refer", "decline"]
    reasons: list[Reason] = Field(
        default_factory=list,
        description="What raised this customer's risk most. Given for refer "
                    "and decline; a decline must state its reasons "
                    "(Regulation B, 12 CFR 1002.9).")
    model_version: str = Field(description="Hash of the model file that decided")
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    trained_at: str | None = None
