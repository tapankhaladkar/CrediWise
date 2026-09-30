# Audit register

A full review of the repository at commit `938fefc` (the original
notebook-only version), with every finding reproduced against the data before
being recorded. Severities are as assessed at audit time. "Closed by" points at
the code that resolves the finding.

Headline: the reported Random Forest accuracy of **79.7% was inflated by ~10
points** by a train/test split that shared customers across the boundary. The
honest figure is ~70%.

---

## Summary

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Outlier removal is a silent no-op | High | Closed |
| 2 | Outliers pervasive; only `Age` attempted | High | Closed |
| 3 | Recorded VIF output is mathematically impossible | Medium | Closed |
| 4 | VIF step drops no features | Medium | Closed |
| 5 | Imputation crosses customer boundaries and precedes the split | Medium | Closed |
| 6 | `Month` label-encoded alphabetically | Medium | Closed |
| 7 | Nominal categoricals label-encoded, not one-hot | Medium | Closed |
| 8 | One third of the grid search failed silently | Medium | Closed |
| 9 | No random seeds on tree models | Medium | Closed |
| 10 | Notebook does not run on current libraries | Medium | Closed |
| 11 | **Train/test leakage inflates accuracy ~10 points** | **Critical** | **Closed** |
| 12 | No baseline, so reported lift is unanchored | High | Closed |
| 13 | Accuracy is the only metric on imbalanced classes | High | Closed |
| 14 | Single split, no cross-validation | Medium | Closed |
| 15 | `Credit_Mix` is a circular feature | Medium | Closed |
| 16 | Two informative features discarded, not parsed | Medium | Closed (partly revised) |
| 17 | SSN- and name-shaped PII committed to the repo | High | Closed |
| 18 | No fairness, explainability or calibration | High | Closed |
| 19 | No data provenance or dictionary | Medium | Closed |
| 20 | README describes a project that does not exist | High | Closed |
| 21 | Two stacked READMEs | Low | Closed |
| 22 | Missing every standard repo file | Medium | Closed |
| 23 | 31 MB CSV committed to git | Medium | Partly closed |
| 24 | Notebook-only, no reusable code | Medium | Closed |
| 25 | No model persistence or inference path | Medium | Closed |
| 26 | Three commits, all untraceable | Low | Closed going forward |

---

## Critical

### 11. Train/test leakage inflates accuracy by ~10 points

The dataset is panel data: **12,500 customers x 8 months = 100,000 rows**,
exactly 8 rows per customer with no exceptions. The notebook drops
`Customer_ID` at cell 3, but dropping the column does not remove the
structure. A random row-level split then scatters each customer's 8 months
across train and test:

```
customers in train: 12,500
customers in test:  10,449
overlap:            10,449   ->  100.0% of test customers seen in training
```

Customers remain trivially identifiable from the surviving features:

| Feature | Constant across a customer's 8 months |
|---|---|
| `Outstanding_Debt` | 92.2% of customers |
| `Num_Bank_Accounts` | 87.8% |
| `Interest_Rate` | 84.7% |
| `Num_Credit_Card` | 81.0% |
| `Total_EMI_per_month` | 72.6% |

And **41.7% of customers carry the same `Credit_Score` in all 8 months**, so
for those, recognising the customer *is* the answer.

Re-running the notebook's own pipeline with `GroupShuffleSplit` on
`Customer_ID` as the only change:

| Model | Row-level split | Grouped split | Delta |
|---|---|---|---|
| Logistic Regression | 61.85% | 62.08% | +0.23 |
| Decision Tree | 69.86% | 60.04% | **-9.82** |
| Random Forest | **79.88%** | **70.03%** | **-9.85** |
| Majority baseline | 53.00% | 52.90% | - |

The asymmetry is diagnostic: logistic regression is unaffected because it
lacks the capacity to memorise, while both tree models lose ~10 points. That
is the signature of memorisation rather than signal.

**Closed by** `src/crediwise/splits.py`, which provides only grouped splitters
and an `assert_no_group_overlap` guard. `tests/test_splits.py` fails the build
if a random split is ever reintroduced.

---

## High severity

### 1. Outlier removal is a silent no-op

Notebook cell 13:

```python
data = df[(df.Age >= Q1 - 1.5*IQR) & (df.Age <= Q3 + 1.5*IQR)]
```

The filtered frame is bound to `data`, which is never referenced again. `df`
— the frame every model trains on — is untouched.

```
rows the notebook believed it dropped: 2,781
rows actually dropped:                     0
df.Age after "outlier removal":  min -500, max 8698
```

**Closed by** `data.apply_domain_ranges`, applied inside `data.clean`.

### 2. Outliers pervasive; only `Age` attempted

| Column | Min | Max | Rejected by the new domain check |
|---|---|---|---|
| `Age` | -500 | 8,698 | 8,482 |
| `Num_of_Loan` | -100 | 1,496 | 4,345 |
| `Num_Credit_Card` | 0 | 1,499 | 2,263 |
| `Interest_Rate` | 1 | 5,797 | 2,034 |
| `Num_of_Delayed_Payment` | -3 | 4,397 | 1,368 |
| `Num_Bank_Accounts` | -1 | 1,798 | 1,335 |
| `Annual_Income` | 7,006 | 24,198,062 | 961 |
| `Delay_from_due_date` | -5 | 67 | 591 |

21,379 corrupt values in total, all previously trained on.

**Closed by** `config.DOMAIN_RANGES` + `data.apply_domain_ranges`.

### 12. No baseline

Always predicting "Standard" scores **52.9%**. The README presented 79.7%
against an implicit baseline of zero. **Closed by** `evaluate.evaluate`, which
always reports `baseline_accuracy` and `lift_over_baseline`.

### 13. Accuracy is the only metric

Class distribution is Standard 53.2% / Poor 29.0% / Good 17.8%. Under a valid
split the minority "Good" class scores only 0.60 F1 against a 0.70 headline —
1,345 genuinely creditworthy customers misclassified as Standard, the costliest
error type in lending. **Closed by** `evaluate.evaluate` (per-class precision,
recall, F1, AUC, Gini, KS, Brier, ECE, confusion matrix).

### 17. PII-shaped data committed to the repo

`credit_score.csv` carried `Name` (10,139 distinct) and `SSN` (12,501
distinct). **88.6% of the SSN strings are structurally valid** — plausible
area, group and serial codes — so they are not dismissible as obviously fake.
The data is near-certainly the Kaggle synthetic credit-score set, but nothing
in the repository said so, leaving it indistinguishable from a real leak.

**Closed by** `scripts/strip_pii.py`; the working copy is `data/credit_score.csv`
with both columns removed, and provenance is stated in `docs/DATA_SHEET.md`.

### 18. No fairness, explainability or calibration

See `docs/FAIRNESS.md` and `docs/MODEL_CARD.md`. **Closed by**
`src/crediwise/fairness.py`, `explain.py`, `policy.py` and the calibration
path in `model.fit_calibrated`.

### 20. README describes a project that does not exist

| README claim | Reality at audit time |
|---|---|
| "Applied One-Hot Encoding" | Never used; `LabelEncoder` only |
| "Feature Selection (VIF) ensuring VIF < 5" | Printed, never applied |
| "Outlier detection & removal" | No-op, zero rows removed |
| `pip install -r requirements.txt` | No such file |
| `git clone .../your-username/...` | Placeholder URL |
| "Random Forest 79.7%" | ~70.0% under a valid split |
| "tuning improved 69.7% -> 70.93%" | Unseeded; not established |
| "100,000 records" | 12,500 people x 8 months |
| "28 financial attributes" | 21 features actually used |

Three of these fail if a reader follows them. **Closed by** the rewritten
README.

---

## Medium severity

### 3. Recorded VIF output is mathematically impossible

The notebook's stored output shows VIFs of `0.0245`, `0.300`, `0.978`. **VIF is
bounded below by 1.0 by definition.** The cause is calling
`variance_inflation_factor` without a constant term. Recomputed correctly, true
VIFs range **1.00 to 3.05** (`Credit_Mix` highest) — so the stated conclusion
happened to hold, but was drawn from invalid numbers.

### 4. VIF step drops no features

No feature is removed on the basis of the VIF table; it is printed and ignored.
Since all true VIFs are below 5, none would be dropped anyway — the step is
decorative. **Closed by** removing it from the modelling path; collinearity is
documented in the model card instead.

### 5. Imputation crosses customer boundaries and precedes the split

`ffill().bfill()` copies down raw row order. When a customer's *first* row is
missing, the value comes from a different person:

```
Monthly_Inhand_Salary    1,861 rows filled from a different customer
Num_of_Delayed_Payment     843
Amount_invested_monthly    565
Num_Credit_Inquiries       236
Monthly_Balance            134
```

It also ran before the split, letting test rows inform training. **Closed by**
moving imputation into the `Pipeline` (`SimpleImputer`), fit on training folds
only. Median imputation was chosen over within-customer fill because it is
reproducible at serving time for an applicant with no history.

### 6. `Month` label-encoded alphabetically

`LabelEncoder` sorts as strings: April=0, August=1, February=2, January=3.
Time order is destroyed, and the Random Forest still assigned it 3.5%
importance — learning the dataset's row layout. **Closed by** dropping `Month`;
measured cost 0.0pp.

### 7. Nominal categoricals label-encoded

`Occupation` and `Payment_Behaviour` received integer codes implying an
ordering (Accountant=0 < Architect=1). **Closed by** `OneHotEncoder` in
`features.build_preprocessor`.

### 8. One third of the grid search failed silently

```
800 fits failed out of a total of 2400
InvalidParameterError: 'max_features' ... Got 'auto' instead.
```

`max_features='auto'` was removed from scikit-learn. The warning was emitted
and not acted on. **Closed by** removing the dead parameter.

### 9. No random seeds on tree models

`DecisionTreeClassifier()` and `RandomForestClassifier()` were unseeded, so
"tuning improved 69.7% -> 70.93%" compared one random draw to another. A 1.2
point gap is within seed variance. **Closed by** `config.RANDOM_STATE`
threaded through every estimator and splitter; `tests/test_pipeline.py`
asserts determinism.

### 10. Notebook does not run on current libraries

Verified on pandas 3.0.6 / scikit-learn 1.9.1:

- `df.fillna(method="ffill")` -> `TypeError` (removed in pandas 3.0). **Hard
  stop at cell 10** — nothing after it executes.
- `max_features='auto'` -> `InvalidParameterError`.

**Closed by** pinned `requirements.txt` and a rewritten pipeline.

### 14. Single split, no cross-validation

Every reported figure was one 80/20 draw with no error bar. **Closed by**
`StratifiedGroupKFold` 5-fold CV reporting mean +/- std.

### 15. `Credit_Mix` is a circular feature

`Credit_Mix` is a bureau-assigned assessment of credit quality and the third
most important feature — partly predicting a credit rating from a credit
rating. Measured cost of removing it: **-1.4pp accuracy, -2.6 macro-F1**.

**Closed by** declaring it a bureau input in `docs/FEATURE_CONTRACT.md` and
shipping a `--no-bureau` variant so the dependency is quantified, not hidden.

### 16. Two informative features discarded rather than parsed — *partly revised*

Both were dropped for being awkward strings. Measured under a grouped split:

| Change | Accuracy | Macro-F1 |
|---|---|---|
| Cleaned baseline | 70.27% | 68.27 |
| **+ `Credit_History_Age` -> months** | **70.58%** | **68.59** |
| + `Type_of_Loan` multi-hot | 69.94% | 67.96 |
| + both | 70.03% | 68.05 |

The original finding assumed both were worth recovering. Only
`Credit_History_Age` is: `Type_of_Loan` adds 8 sparse indicators that dilute
the signal and **costs 0.33pp**. It stays dropped — now on evidence rather
than convenience.

### 19. No data provenance or dictionary

**Closed by** `docs/DATA_SHEET.md`.

### 22. Missing every standard repo file

No `requirements.txt`, `.gitignore`, `LICENSE`, tests or CI; `git ls-files`
returned exactly three files. **Closed by** Phase 0.

### 23. 31 MB CSV committed to git — *partly closed*

The working copy is now the PII-stripped `data/credit_score.csv`. **The
original file remains in git history**; fully purging it requires a history
rewrite (`git filter-repo`), which is cheap at this commit count but has not
been done because it rewrites published hashes. Tracked as open.

### 24. Notebook-only, no reusable code

35-line straight-line preprocessing cell, nothing importable or testable.
**Closed by** the `src/crediwise/` package.

### 25. No model persistence or inference path

Nothing was saved — not even the `LabelEncoder` mappings, so the notebook's
model could never have scored a new applicant. **Closed by**
`artifacts/crediwise_model.joblib` (preprocessing, forest, calibrator and
policy in one bundle) and the FastAPI service.

---

## Low severity

### 21. Two stacked READMEs

The file opened with `# CrediWise` then immediately `# Credit Score
Classification Project` — two H1s, two project names. **Closed by** the
rewrite.

### 26. Three commits, all untraceable

`Initial commit` -> `Updated the dataset and code` (101,399 lines) -> `Update
README.md`. **Closed going forward**; this rebuild is committed in reviewable
phases.
