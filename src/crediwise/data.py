"""Loading, cleaning and feature engineering.

The source data is unusually clean -- no missing values, no impossible
numbers -- so cleaning is mostly about making it readable and being explicit
about the codes the documentation does not explain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or cfg.DATA_PATH)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Readable names, labelled categories, duplicates removed.

    - The repayment status codes are kept as numbers. The documentation
      defines -1 as "paid duly" and 1-9 as months of delay, but the two most
      common values in the data, -2 and 0, are undocumented. They sort
      sensibly (no balance < paid duly < revolving < late) and the tree model
      treats them as ordered thresholds, so no meaning is invented for them.
    - Negative bill amounts are credit balances from overpayment, not errors,
      and are kept.
    - 35 records appear twice with identical values and outcomes (mostly
      inactive accounts with all-zero bills). One copy of each is dropped so
      no record can sit in both the training and test sets.
    """
    df = raw.rename(columns=cfg.RENAME).drop(columns=["ID"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    df["sex"] = df["sex"].map(cfg.SEX_LABELS).fillna("unknown")
    df["education"] = df["education"].map(cfg.EDUCATION_LABELS).fillna("unknown")
    df["marital_status"] = df["marital_status"].map(cfg.MARITAL_LABELS).fillna("unknown")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summaries of the six-month history.

    Used both in training and by the web service on a single applicant, so
    a feature is computed identically in both places.
    """
    out = df.copy()
    out["utilization"] = out["bill_sep"] / out["credit_limit"]
    out["avg_utilization"] = out[cfg.BILLS].mean(axis=1) / out["credit_limit"]
    out["months_late"] = (out[cfg.STATUS] >= 1).sum(axis=1)
    out["worst_status"] = out[cfg.STATUS].max(axis=1)
    # A payment made in September settles the August statement. Where nothing
    # was owed, treat the bill as fully paid. Capped at 2 so a large
    # prepayment does not dominate.
    owed = out["bill_aug"].clip(lower=0)
    out["paid_share"] = np.where(
        owed > 0, (out["paid_sep"] / owed.where(owed > 0)).clip(upper=2), 1.0
    )
    return out


def add_audit_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    edges = [cfg.AGE_BANDS[0][0] - 1] + [high for _, high in cfg.AGE_BANDS]
    labels = [f"{low}-{high}" for low, high in cfg.AGE_BANDS]
    out["age_band"] = pd.cut(out["age"], bins=edges, labels=labels).astype("object")
    return out


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Everything the pipeline needs, in one frame."""
    return add_audit_dimensions(add_features(clean(load_raw(path))))
