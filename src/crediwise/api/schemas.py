"""Request and response contracts for the scoring API.

The field list here *is* the feature contract: anything the model needs at
scoring time must be suppliable by a caller. Fields the source dataset carries
but the model does not use (Age, Month, Type_of_Loan, the direct identifiers)
are deliberately absent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    """One applicant-month. All fields optional -- missing values are imputed
    with the medians learned at training time, so partial records still score.
    """

    Annual_Income: float | None = Field(None, ge=0, le=1_000_000)
    Monthly_Inhand_Salary: float | None = Field(None, ge=0, le=100_000)
    Num_Bank_Accounts: float | None = Field(None, ge=0, le=20)
    Num_Credit_Card: float | None = Field(None, ge=0, le=20)
    Interest_Rate: float | None = Field(None, ge=0, le=50)
    Num_of_Loan: float | None = Field(None, ge=0, le=20)
    Delay_from_due_date: float | None = Field(None, ge=0, le=200)
    Num_of_Delayed_Payment: float | None = Field(None, ge=0, le=100)
    Changed_Credit_Limit: float | None = None
    Num_Credit_Inquiries: float | None = Field(None, ge=0, le=50)
    Outstanding_Debt: float | None = Field(None, ge=0, le=500_000)
    Credit_Utilization_Ratio: float | None = Field(None, ge=0, le=100)
    Total_EMI_per_month: float | None = Field(None, ge=0, le=100_000)
    Amount_invested_monthly: float | None = Field(None, ge=0, le=100_000)
    Monthly_Balance: float | None = None
    Credit_History_Months: float | None = Field(
        None, ge=0, le=600, description="Length of credit history in months"
    )
    Credit_Mix: Literal[0, 1, 2] | None = Field(
        None, description="Bureau-supplied: 0=Bad, 1=Standard, 2=Good"
    )
    Payment_of_Min_Amount: Literal[0, 1] | None = None
    Occupation: str | None = None
    Payment_Behaviour: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Annual_Income": 19114.12,
                "Monthly_Inhand_Salary": 1824.84,
                "Num_Bank_Accounts": 3,
                "Num_Credit_Card": 4,
                "Interest_Rate": 3,
                "Num_of_Loan": 4,
                "Delay_from_due_date": 3,
                "Num_of_Delayed_Payment": 7,
                "Changed_Credit_Limit": 11.27,
                "Num_Credit_Inquiries": 4,
                "Outstanding_Debt": 809.98,
                "Credit_Utilization_Ratio": 26.82,
                "Total_EMI_per_month": 49.57,
                "Amount_invested_monthly": 80.42,
                "Monthly_Balance": 312.49,
                "Credit_History_Months": 265,
                "Credit_Mix": 2,
                "Payment_of_Min_Amount": 0,
                "Occupation": "Scientist",
                "Payment_Behaviour": "High_spent_Small_value_payments",
            }]
        }
    }


class ReasonCode(BaseModel):
    feature: str
    reason: str
    contribution: float


class ScoreResponse(BaseModel):
    risk_band: str = Field(description="Poor, Standard or Good")
    probabilities: dict[str, float]
    decision: str = Field(description="approve, refer or decline")
    is_adverse: bool
    cost_optimal_band: str
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Principal factors driving an adverse outcome "
                    "(ECOA 12 CFR 1002.9). Populated when is_adverse is true.",
    )
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    trained_at: str | None = None
