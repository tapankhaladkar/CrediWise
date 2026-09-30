"""Regression tests for the leakage defect (docs/AUDIT.md #11)."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from crediwise import config as cfg
from crediwise import data as data_mod
from crediwise import splits


def test_holdout_split_is_customer_disjoint(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    train_idx, test_idx = splits.holdout_split(cleaned)
    splits.assert_no_group_overlap(cleaned[cfg.GROUP_COL], train_idx, test_idx)


def test_cv_folds_are_customer_disjoint(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    splitter = splits.cv_splitter(n_splits=3)
    groups = cleaned[cfg.GROUP_COL]
    for train_idx, test_idx in splitter.split(
        cleaned, cleaned[cfg.TARGET_COL], groups=groups
    ):
        splits.assert_no_group_overlap(groups, train_idx, test_idx)


def test_random_split_is_detected_as_leaky(raw_sample):
    """The guard must actually fire on the original notebook's split."""
    cleaned = data_mod.clean(raw_sample)
    train_idx, test_idx = train_test_split(
        np.arange(len(cleaned)), test_size=0.2, random_state=42
    )
    with pytest.raises(AssertionError, match="both train and test"):
        splits.assert_no_group_overlap(cleaned[cfg.GROUP_COL], train_idx, test_idx)
