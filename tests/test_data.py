import numpy as np
import pandas as pd

from crediwise import config as cfg
from crediwise import data as data_mod


def test_placeholder_tokens_become_missing(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    assert cleaned["Occupation"].isna().any(), "'_______' should read as missing"
    assert cleaned["Credit_Mix"].isna().any(), "'_' should read as missing"
    assert cleaned["Payment_of_Min_Amount"].isna().any(), "'NM' should read as missing"
    assert cleaned["Payment_Behaviour"].isna().any(), "'!@9#%8' should read as missing"


def test_underscore_suffixes_are_stripped(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    assert cleaned["Annual_Income"].notna().sum() > 0
    assert pd.api.types.is_numeric_dtype(cleaned["Annual_Income"])


def test_impossible_values_rejected(raw_sample):
    """Audit #1/#2: the notebook trained on ages of -500 and rates of 5797%."""
    cleaned = data_mod.clean(raw_sample)
    assert cleaned["Age"].min() >= 18
    assert cleaned["Age"].max() <= 100
    assert cleaned["Interest_Rate"].max() <= 50
    assert cleaned["Num_Bank_Accounts"].max() <= 20


def test_credit_history_parses_to_months():
    """Audit #16: the notebook dropped this column rather than parse it."""
    series = pd.Series(["22 Years and 1 Months", "3 Years and 0 Months", "NA", None])
    parsed = data_mod.parse_credit_history(series)
    assert parsed.iloc[0] == 22 * 12 + 1
    assert parsed.iloc[1] == 36
    assert np.isnan(parsed.iloc[2]) and np.isnan(parsed.iloc[3])


def test_feature_frame_excludes_protected_and_identifiers(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    X = data_mod.feature_frame(cleaned)
    for banned in ("Age", "Month", "Customer_ID", "ID", "Credit_Score"):
        assert banned not in X.columns, f"{banned} must not be a model input"


def test_no_bureau_variant_drops_credit_mix(raw_sample):
    cleaned = data_mod.clean(raw_sample)
    assert "Credit_Mix" in data_mod.feature_frame(cleaned, include_bureau=True)
    assert "Credit_Mix" not in data_mod.feature_frame(cleaned, include_bureau=False)
