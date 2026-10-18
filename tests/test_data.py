import numpy as np
import pandas as pd

from crediwise import config as cfg
from crediwise import data as data_mod


def test_columns_get_readable_names(raw_sample):
    df = data_mod.clean(raw_sample)
    assert "ID" not in df.columns
    assert {"credit_limit", "status_sep", "bill_apr", "paid_aug", "defaulted"} <= set(df.columns)
    assert "PAY_0" not in df.columns


def test_duplicate_records_are_dropped(raw_sample):
    """A record in both train and test would flatter the test score."""
    df = data_mod.clean(raw_sample)
    assert len(df) == len(raw_sample) - 1
    assert not df.duplicated().any()


def test_undocumented_codes_become_unknown_not_guesses(raw_sample):
    df = data_mod.clean(raw_sample)
    assert set(df["education"]) <= {"graduate_school", "university", "high_school",
                                    "other", "unknown"}
    assert "unknown" in set(df["education"])        # codes 0, 5, 6
    assert "unknown" in set(df["marital_status"])   # code 0


def test_negative_bills_are_kept(raw_sample):
    """Overpayment leaves a credit balance; it is not an error."""
    assert (data_mod.clean(raw_sample)["bill_sep"] < 0).any()


def test_engineered_features():
    row = pd.DataFrame([{
        "credit_limit": 10_000,
        **{s: 0 for s in cfg.STATUS}, "status_sep": 2, "status_jul": 1,
        **{b: 1_000 for b in cfg.BILLS}, "bill_sep": 5_000,
        **{p: 0 for p in cfg.PAID}, "paid_sep": 500,
    }])
    out = data_mod.add_features(row).iloc[0]
    assert out["utilization"] == 0.5
    assert out["months_late"] == 2
    assert out["worst_status"] == 2
    assert out["paid_share"] == 0.5                 # paid 500 of the 1,000 owed


def test_paid_share_edge_cases():
    base = {"credit_limit": 1, **{s: 0 for s in cfg.STATUS},
            **{b: 0 for b in cfg.BILLS}, **{p: 0 for p in cfg.PAID}}
    rows = pd.DataFrame([
        base | {"bill_aug": 0, "paid_sep": 0},       # nothing owed -> treated as paid
        base | {"bill_aug": -300, "paid_sep": 0},    # in credit -> treated as paid
        base | {"bill_aug": 100, "paid_sep": 900},   # big prepayment -> capped at 2
    ])
    assert list(data_mod.add_features(rows)["paid_share"]) == [1.0, 1.0, 2.0]


def test_model_inputs_exclude_protected_attributes():
    for banned in ["sex", "marital_status", "age", "education", "defaulted"]:
        assert banned not in cfg.FEATURES


def test_full_dataset_loads_without_gaps():
    df = data_mod.load_dataset()
    assert len(df) == 29_965
    assert not df[cfg.FEATURES].isna().any().any()
