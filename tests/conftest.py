import numpy as np
import pandas as pd
import pytest

from crediwise import config as cfg


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """A small synthetic frame shaped like the real extract, dirt included."""
    rng = np.random.default_rng(0)
    n_customers, months = 40, 8
    rows = []
    for c in range(n_customers):
        cid = f"CUS_{c:04d}"
        income = float(rng.uniform(10_000, 90_000))
        for m, name in enumerate(
            ["January", "February", "March", "April", "May", "June", "July", "August"]
        ):
            rows.append({
                "ID": f"0x{c*months+m:04x}",
                "Customer_ID": cid,
                "Month": name,
                "Age": -500 if (c + m) % 17 == 0 else int(rng.integers(20, 70)),
                "Occupation": "_______" if m == 2 else "Scientist",
                "Annual_Income": f"{income}_" if m == 1 else income,
                "Monthly_Inhand_Salary": np.nan if m == 3 else income / 12,
                "Num_Bank_Accounts": 1798 if (c + m) % 23 == 0 else int(rng.integers(1, 8)),
                "Num_Credit_Card": int(rng.integers(1, 8)),
                "Interest_Rate": 5797 if (c + m) % 29 == 0 else int(rng.integers(1, 30)),
                "Num_of_Loan": str(int(rng.integers(0, 6))),
                "Type_of_Loan": "Auto Loan, and Home Equity Loan",
                "Delay_from_due_date": int(rng.integers(0, 40)),
                "Num_of_Delayed_Payment": str(int(rng.integers(0, 20))),
                "Changed_Credit_Limit": "_" if m == 4 else round(float(rng.uniform(0, 30)), 2),
                "Num_Credit_Inquiries": float(rng.integers(0, 12)),
                "Credit_Mix": "_" if m == 5 else rng.choice(["Bad", "Standard", "Good"]),
                "Outstanding_Debt": round(float(rng.uniform(0, 4000)), 2),
                "Credit_Utilization_Ratio": round(float(rng.uniform(20, 45)), 4),
                "Credit_History_Age": f"{int(rng.integers(1, 30))} Years and {m} Months",
                "Payment_of_Min_Amount": "NM" if m == 6 else rng.choice(["Yes", "No"]),
                "Total_EMI_per_month": round(float(rng.uniform(0, 400)), 4),
                "Amount_invested_monthly": round(float(rng.uniform(0, 300)), 4),
                "Payment_Behaviour": "!@9#%8" if m == 7 else "High_spent_Small_value_payments",
                "Monthly_Balance": round(float(rng.uniform(100, 600)), 4),
                "Credit_Score": rng.choice(["Poor", "Standard", "Good"]),
            })
    return pd.DataFrame(rows)
