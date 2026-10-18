import numpy as np
import pandas as pd
import pytest


def _raw_frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """A small frame in the exact source format, quirks included.

    Default is made more likely by late payments so a model has signal to
    find. Undocumented codes, a negative bill and a duplicate record are
    planted so the cleaning code is exercised.
    """
    rng = np.random.default_rng(seed)
    limit = rng.choice([20_000, 50_000, 120_000, 300_000], size=n)
    lateness = rng.choice([-2, -1, 0, 0, 1, 2, 3], size=(n, 6))
    bills = (rng.uniform(0, 1, size=(n, 6)) * limit[:, None]).round()
    paid = (rng.uniform(0, 0.3, size=(n, 6)) * bills).round()
    risk = 1 / (1 + np.exp(-(lateness.clip(min=0).sum(axis=1) - 3)))
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": limit,
        "SEX": rng.choice([1, 2], size=n),
        "EDUCATION": rng.choice([1, 2, 3, 4, 0, 5, 6], size=n,
                                p=[.3, .4, .2, .04, .02, .02, .02]),
        "MARRIAGE": rng.choice([1, 2, 3, 0], size=n, p=[.45, .5, .03, .02]),
        "AGE": rng.integers(21, 75, size=n),
    })
    for i, name in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        df[name] = lateness[:, i]
    for i in range(6):
        df[f"BILL_AMT{i + 1}"] = bills[:, i]
    for i in range(6):
        df[f"PAY_AMT{i + 1}"] = paid[:, i]
    df["default payment next month"] = (rng.uniform(size=n) < risk).astype(int)

    df.loc[0, "BILL_AMT1"] = -5_000                 # credit balance, legitimate
    dup = df.iloc[[1]].copy()
    dup["ID"] = n + 1                               # same record, new ID
    return pd.concat([df, dup], ignore_index=True)


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    return _raw_frame()
