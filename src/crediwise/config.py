"""Schema, domain ranges and the feature contract.

Everything the rest of the package needs to know about the shape of the data
lives here, so the training pipeline and the serving API cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "credit_score.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

#: Column holding the panel key. Each customer contributes 8 monthly rows, so
#: every split in this project must be grouped on it -- see docs/AUDIT.md #11.
GROUP_COL = "Customer_ID"
TARGET_COL = "Credit_Score"

#: Ordinal target. Higher is better credit standing.
TARGET_MAP = {"Poor": 0, "Standard": 1, "Good": 2}
TARGET_LABELS = {v: k for k, v in TARGET_MAP.items()}

#: Columns dropped before modelling, with the reason. Kept as data rather than
#: a bare list so the model card can render it.
DROPPED_COLUMNS = {
    "ID": "row identifier, no predictive content",
    "Customer_ID": "panel key -- used for grouping, never as a feature",
    "Month": "artifact of the panel layout; costs nothing to drop (audit #6)",
    "Age": "ECOA-protected basis; removal measured at -0.02pp (noise), "
           "retained only as a fairness audit dimension (audit #18)",
    "Type_of_Loan": "multi-hot encoding measured at -0.33pp macro-F1 during "
                    "the audit; dropped on evidence (audit #16)",
}

#: Values that stand in for "missing" in the raw file. Each was found by
#: inspection of the source data; together they cover 49,955 cells.
PLACEHOLDER_TOKENS = {
    "Occupation": ["_______"],
    "Payment_Behaviour": ["!@9#%8"],
    "Credit_Mix": ["_"],
    "Payment_of_Min_Amount": ["NM"],
    "Changed_Credit_Limit": ["_"],
}

#: Plausible ranges for numeric fields. Values outside these are treated as
#: corrupt and set to NaN for imputation rather than silently trained on.
#: The source data contains ages of -500 and 8698, interest rates of 5797%
#: and 1798 bank accounts (audit #2).
DOMAIN_RANGES = {
    "Age": (18, 100),
    "Annual_Income": (0, 1_000_000),
    "Monthly_Inhand_Salary": (0, 100_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 50),
    "Num_of_Loan": (0, 20),
    "Delay_from_due_date": (0, 200),
    "Num_of_Delayed_Payment": (0, 100),
    "Num_Credit_Inquiries": (0, 50),
    "Outstanding_Debt": (0, 500_000),
    "Credit_Utilization_Ratio": (0, 100),
    "Total_EMI_per_month": (0, 100_000),
    "Amount_invested_monthly": (0, 100_000),
    "Monthly_Balance": (-10_000, 1_000_000),
    "Credit_History_Months": (0, 600),
}

#: Ordinal categoricals, mapped to integers that preserve their real order.
ORDINAL_MAPS = {
    "Credit_Mix": {"Bad": 0, "Standard": 1, "Good": 2},
    "Payment_of_Min_Amount": {"No": 0, "Yes": 1},
}

#: Nominal categoricals. One-hot encoded -- LabelEncoder would impose a false
#: ordering (audit #7).
NOMINAL_FEATURES = ["Occupation", "Payment_Behaviour"]

NUMERIC_FEATURES = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_History_Months",
    "Credit_Mix",
    "Payment_of_Min_Amount",
]

#: Features sourced from a credit bureau rather than the applicant or the
#: lender's own records. `Credit_Mix` is a bureau-assigned assessment of credit
#: quality, which makes predicting a credit band from it partly circular
#: (audit #15). The model keeps it, but `--no-bureau` trains a variant without
#: it so the dependency is quantified rather than hidden.
BUREAU_FEATURES = ["Credit_Mix"]

#: Protected and proxy attributes for the fairness audit. These are never model
#: inputs. See docs/FAIRNESS.md for what this dataset cannot test.
AUDIT_DIMENSIONS = ["age_band", "Occupation", "income_quintile"]

AGE_BANDS = [(18, 25), (26, 35), (36, 45), (46, 55), (56, 100)]
