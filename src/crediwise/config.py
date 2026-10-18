"""Paths, column names and the feature list.

Everything the training pipeline and the web service need to agree on lives
here, so the two cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "credit_default.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "crediwise_model.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
SEARCH_ITERATIONS = 40

#: The six months of history, most recent first. The source data covers
#: April-September 2005 and the outcome is October 2005.
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]

#: Source column names -> readable names. The source calls the most recent
#: repayment status PAY_0 and skips PAY_1, which is easy to misread.
RENAME = {
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marital_status",
    "AGE": "age",
    "default payment next month": "defaulted",
    **{src: f"status_{m}" for src, m in
       zip(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], MONTHS)},
    **{f"BILL_AMT{i}": f"bill_{m}" for i, m in enumerate(MONTHS, start=1)},
    **{f"PAY_AMT{i}": f"paid_{m}" for i, m in enumerate(MONTHS, start=1)},
}

TARGET = "defaulted"

STATUS = [f"status_{m}" for m in MONTHS]
BILLS = [f"bill_{m}" for m in MONTHS]
PAID = [f"paid_{m}" for m in MONTHS]

#: What a lender holds for each customer: the limit it granted plus six months
#: of statements, payments and repayment status.
RAW_FEATURES = ["credit_limit"] + STATUS + BILLS + PAID

#: Summaries computed from the raw history (data.add_features). They add
#: little accuracy to the boosted model (+0.002 CV AUC, within noise) but help
#: a linear model a lot (+0.032), and they let declines cite plain reasons such
#: as "months paid late" instead of a single month's status code.
ENGINEERED_FEATURES = [
    "utilization",
    "avg_utilization",
    "months_late",
    "worst_status",
    "paid_share",
]

#: Model inputs. Credit behaviour only -- see PROTECTED below.
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

#: Never model inputs. Sex, marital status and age are protected under US
#: fair-lending law (ECOA); including them measured at 0.7844 CV AUC against
#: 0.7853 without, so excluding them costs nothing. Education added +0.0001
#: and is a socioeconomic proxy, so it is excluded too. All four are kept for
#: the fairness audit.
PROTECTED = ["sex", "marital_status", "age"]
AUDIT_DIMENSIONS = ["sex", "marital_status", "age_band", "education"]

#: Documented category codes. Codes outside these (education 0/5/6, marital
#: status 0) appear in the data but not in the documentation and are labelled
#: "unknown" rather than guessed at.
SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate_school", 2: "university", 3: "high_school", 4: "other"}
MARITAL_LABELS = {1: "married", 2: "single", 3: "other"}

AGE_BANDS = [(21, 29), (30, 39), (40, 49), (50, 59), (60, 79)]
