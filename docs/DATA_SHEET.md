# Data sheet

## Provenance

Public **synthetic** credit-score classification dataset, widely circulated on
Kaggle. It is not real customer data and describes no real person.

This matters because the original file shipped `Name` and `SSN` columns in
which **88.6% of the SSN strings were structurally valid** — plausible area,
group and serial codes. Without a provenance statement, the file was
indistinguishable from a genuine PII leak to any reader or automated scanner.
Both columns are now removed (`scripts/strip_pii.py`); `data/credit_score.csv`
is the working copy.

The original file remains in this repository's git history (audit #23). Fully
removing it requires `git filter-repo`, which rewrites published commit hashes.

## Shape

- 100,000 rows = **12,500 customers x 8 months** (January-August), exactly 8
  rows per customer.
- 26 columns after identifier removal; 20 reach the model, 18 numeric and 2
  nominal.
- The panel structure is the single most important fact about this dataset.
  Any row-level split leaks — see `docs/AUDIT.md` #11.

## Target

`Credit_Score` in {Poor, Standard, Good}, mapped to {0, 1, 2}.

| Class | Count | Share |
|---|---|---|
| Standard | 53,174 | 53.2% |
| Poor | 28,998 | 29.0% |
| Good | 17,828 | 17.8% |

Always predicting "Standard" scores **52.9%** — the baseline every metric in
this project is reported against.

How the labels were assigned is **not documented by the source**. This is a
real limitation: the ground truth cannot be traced to an underwriting policy
or an observed default outcome.

## Known defects

Five sentinel tokens stand in for missing values across 49,955 cells:

| Column | Token | Count |
|---|---|---|
| `Credit_Mix` | `_` | 20,195 |
| `Payment_of_Min_Amount` | `NM` | 12,007 |
| `Payment_Behaviour` | `!@9#%8` | 7,600 |
| `Occupation` | `_______` | 7,062 |
| `Changed_Credit_Limit` | `_` | 2,091 |

Numeric columns additionally carry stray `_` suffixes and impossible values —
ages of -500 and 8,698, interest rates of 5,797%, 1,798 bank accounts. The
domain ranges in `config.DOMAIN_RANGES` reject 21,379 such values to NaN for
imputation.

## Suitability

Adequate for demonstrating a credit-risk modelling pipeline. **Not adequate**
for: validating a real scoring system, any claim about race or sex fairness
(six of seven ECOA bases are absent — see `docs/FAIRNESS.md`), or inferring
anything about real populations.
