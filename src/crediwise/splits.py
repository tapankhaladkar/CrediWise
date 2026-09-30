"""Grouped splitting.

This dataset is panel data: 12,500 customers contribute exactly 8 monthly rows
each. A random row-level split puts the same customer in train and test, and
because five features are near-constant within a customer (``Outstanding_Debt``
is identical across all 8 months for 92.2% of them) the model can identify the
customer rather than learn credit risk. On the original notebook's split, 100%
of test customers also appeared in training, inflating Random Forest accuracy
from a true ~70% to a reported 79.7% (audit #11).

Every split in this project therefore groups on ``Customer_ID``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import config as cfg


def holdout_split(
    df: pd.DataFrame,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Customer-disjoint train/test indices."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    return next(splitter.split(df, df[cfg.TARGET_COL], groups=df[cfg.GROUP_COL]))


def cv_splitter(
    n_splits: int = cfg.N_SPLITS, random_state: int = cfg.RANDOM_STATE
) -> StratifiedGroupKFold:
    """Cross-validation that is both grouped and class-stratified."""
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )


def assert_no_group_overlap(
    groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Fail loudly if a split leaks a customer across the boundary."""
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise AssertionError(
            f"{len(overlap)} customers appear in both train and test "
            f"(e.g. {sorted(overlap)[:3]}). This is the defect described in "
            f"docs/AUDIT.md #11."
        )
