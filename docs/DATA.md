# Data

## Source

**Default of Credit Card Clients**, UCI Machine Learning Repository.
Yeh, I. (2009). https://doi.org/10.24432/C55S3H
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

Real, anonymised records of 30,000 credit card customers of a bank in Taiwan.
Each row holds one customer's credit limit, some personal details, and six
months of history from **April to September 2005**. The outcome is whether
they missed their payment in **October 2005**.

`data/credit_default.csv` is a direct conversion of the published spreadsheet,
with the original column names. `make data` (`scripts/download_data.py`)
re-downloads it and refuses to continue unless the archive matches SHA-256
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`, so the
results can't silently change underneath a re-run.

## Columns

Renamed in `src/crediwise/data.py` for readability; the CSV is untouched.

| Name used here | Source column | Meaning | Model input? |
|---|---|---|---|
| `credit_limit` | `LIMIT_BAL` | Credit granted, NT dollars | Yes |
| `status_sep` … `status_apr` | `PAY_0`, `PAY_2` … `PAY_6` | Repayment status per month (codes below) | Yes |
| `bill_sep` … `bill_apr` | `BILL_AMT1` … `BILL_AMT6` | Statement balance per month | Yes |
| `paid_sep` … `paid_apr` | `PAY_AMT1` … `PAY_AMT6` | Amount paid during the month | Yes |
| `sex` | `SEX` | 1 male, 2 female | **No**: fairness audit only |
| `marital_status` | `MARRIAGE` | 1 married, 2 single, 3 other | **No**: fairness audit only |
| `age` | `AGE` | Years, 21–79 | **No**: fairness audit only |
| `education` | `EDUCATION` | 1 graduate school, 2 university, 3 high school, 4 other | **No**: fairness audit only |
| `defaulted` | `default payment next month` | 1 if the October payment was missed | Target |

The source names the most recent status column `PAY_0` and skips `PAY_1`,
which is easy to misread; the renaming removes that trap.

Five summary features are computed from the history (`months_late`,
`worst_status`, `utilization`, `avg_utilization`, `paid_share`); see
`data.add_features`.

## Repayment status codes

| Code | Meaning | Source |
|---|---|---|
| −1 | Paid in full | Documented |
| 1 – 9 | That many months late | Documented |
| **−2** | Not defined (looks like "no balance to pay") | **Undocumented** |
| **0** | Not defined (looks like "revolving credit, minimum paid") | **Undocumented** |

The two undocumented codes are the *most common* values: in September, 14,737
customers have status 0 and 2,759 have −2. Rather than invent a meaning, the
model treats all codes as ordered numbers (−2 < −1 < 0 < 1 …), which is how a
tree model uses them anyway.

## Cleaning decisions

The data is unusually clean: no missing values and no impossible numbers.
What remained:

| Issue | Count | Decision |
|---|---|---|
| Duplicate records (identical in every field except ID) | 35 records appearing twice | **Dropped one copy each.** 60 of the 70 rows are inactive accounts with all-zero bills and payments, so plausibly different customers who look identical, and each pair has the same outcome. Keeping both copies could put one in training and its twin in the test set, flattering the score. |
| Negative bill amounts | 3,932 values | **Kept.** A customer who overpays is in credit; that's real, not an error. |
| Undocumented education codes (0, 5, 6) | 345 customers | **Labelled "unknown"**, not merged into a documented group. Education isn't a model input. |
| Undocumented marital status code (0) | 54 customers | **Labelled "unknown"**. Not a model input. |

After cleaning: **29,965 customers, 22.1% defaulted**.

## What the model is not given, and why

Sex, marital status and age are protected bases under US fair-lending law
(ECOA). Measured by cross-validation on the training data, including them
changed AUC from 0.7853 to 0.7844, i.e. nothing, so they are left out.
Education added +0.0001 and is a socioeconomic proxy, so it is left out too.
All four are kept for the fairness audit.

## Limits of this data

- **One lender, one country, 2005.** Card markets, regulation and customer
  behaviour have all changed since.
- **One time window.** Every customer is observed over the same six months, so
  there's no way to train on one period and test on a later one.
- **Missing protected attributes.** Race, religion, national origin and receipt
  of public assistance aren't recorded, so fairness on those can't be checked.
