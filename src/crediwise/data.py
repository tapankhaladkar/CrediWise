"""Loading and cleaning the raw credit-score extract.

The source file is dirty in three distinct ways, each handled separately here:

1. Numeric columns carry stray ``_`` characters ("23_", "19114.12_").
2. Five columns use sentinel strings for missing values ("_______", "!@9#%8").
3. Numeric columns contain impossible values -- ages of -500 and 8698,
   interest rates of 5797%, 1798 bank accounts. These are set to NaN and
   imputed rather than trained on (audit #1, #2).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

_HISTORY_RE = re.compile(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", re.IGNORECASE)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip the stray underscores the export left behind, then coerce."""
    cleaned = series.astype(str).str.replace("_", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_credit_history(series: pd.Series) -> pd.Series:
    """Convert "22 Years and 1 Months" to a month count.

    The original notebook dropped this column for being awkward to parse
    (audit #16). Length of credit history is one of the stronger predictors
    available here: recovering it is worth +0.31pp accuracy.
    """

    def one(value: object) -> float:
        match = _HISTORY_RE.match(str(value))
        if not match:
            return np.nan
        return int(match.group(1)) * 12 + int(match.group(2))

    return series.map(one)


def apply_domain_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set out-of-range numeric values to NaN, reporting how many per column."""
    out = df.copy()
    rejected: dict[str, int] = {}
    for column, (low, high) in cfg.DOMAIN_RANGES.items():
        if column not in out.columns:
            continue
        bad = out[column].notna() & ~out[column].between(low, high)
        if bad.any():
            rejected[column] = int(bad.sum())
        out.loc[bad, column] = np.nan
    return out, rejected


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or cfg.DATA_PATH, low_memory=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw extract into a typed, range-validated frame.

    Returns every modelling feature plus the group key, the target and the
    fairness audit dimensions. Imputation is deliberately *not* done here --
    it belongs inside the pipeline so it is fit on training folds only
    (audit #5).
    """
    df = pd.DataFrame(index=raw.index)

    # Sentinel strings to NaN, before anything else reads these columns.
    for column, tokens in cfg.PLACEHOLDER_TOKENS.items():
        if column in raw.columns:
            raw = raw.copy()
            raw[column] = raw[column].replace(tokens, np.nan)

    numeric_sources = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    for column in numeric_sources:
        df[column] = _to_numeric(raw[column])

    df["Credit_History_Months"] = parse_credit_history(raw["Credit_History_Age"])

    for column, mapping in cfg.ORDINAL_MAPS.items():
        df[column] = raw[column].map(mapping)

    for column in cfg.NOMINAL_FEATURES:
        df[column] = raw[column].astype("object")

    # Audit-only. Age is not a model input; it is the single ECOA-protected
    # basis this dataset carries, so it is kept for the fairness report.
    df["Age"] = _to_numeric(raw["Age"])

    df[cfg.GROUP_COL] = raw[cfg.GROUP_COL]
    if cfg.TARGET_COL in raw.columns:
        df[cfg.TARGET_COL] = raw[cfg.TARGET_COL].map(cfg.TARGET_MAP).astype("int8")

    df, _ = apply_domain_ranges(df)
    return df


def add_audit_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the subgroup columns the fairness report slices on."""
    out = df.copy()
    edges = [cfg.AGE_BANDS[0][0] - 1] + [high for _, high in cfg.AGE_BANDS]
    labels = [f"{low}-{high}" for low, high in cfg.AGE_BANDS]
    out["age_band"] = pd.cut(out["Age"], bins=edges, labels=labels)
    out["income_quintile"] = pd.qcut(
        out["Annual_Income"], 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop"
    )
    return out


def feature_frame(df: pd.DataFrame, include_bureau: bool = True) -> pd.DataFrame:
    """Select exactly the model inputs, in a fixed column order."""
    numeric = [c for c in cfg.NUMERIC_FEATURES
               if include_bureau or c not in cfg.BUREAU_FEATURES]
    return df[numeric + cfg.NOMINAL_FEATURES]
