# CrediWise

Credit score classification and risk assessment, with grouped validation,
probability calibration, SHAP reason codes and a bounded fairness audit.

> **Synthetic data, not a lending system.** Trained on a public synthetic
> dataset whose labels have no documented provenance. Six of the seven ECOA
> protected bases are absent from it, so nothing here clears a model for
> discriminatory impact. See [`docs/FAIRNESS.md`](docs/FAIRNESS.md).

---

## The headline number, and why it is lower than it looks

An earlier version of this project reported **79.7%** Random Forest accuracy.
That figure was wrong, and the reason is worth stating plainly.

The dataset is **panel data**: 12,500 customers contributing exactly 8 monthly
rows each. Dropping `Customer_ID` removes the column but not the structure, so
a random row-level split scattered each customer's 8 months across train and
test — **100% of test customers had been seen in training**. With five
features near-constant within a customer (`Outstanding_Debt` is identical
across all 8 months for 92.2% of them) and 41.7% of customers carrying one
unchanging label, the model was identifying people rather than learning credit
risk.

Re-running the original notebook's own pipeline with the split as the only
change:

| Model | Random row split | Grouped by customer | Delta |
|---|---|---|---|
| Logistic Regression | 61.85% | 62.08% | +0.23 |
| Decision Tree | 69.86% | 60.04% | **-9.82** |
| Random Forest | **79.88%** | **70.03%** | **-9.85** |

Logistic regression is unaffected because it lacks the capacity to memorise;
both tree models lose ~10 points. That asymmetry is the signature of
memorisation.

The size of the inflation depends on how much capacity the model has to
memorise with. Those figures are for fully grown trees, as the notebook used.
The model shipped here is deliberately shallower (`min_samples_leaf=20`), and
its own leakage gap is correspondingly smaller — around 3.5 points, reported
under "Why this differs from the original notebook" in the model card. Both
numbers are real; they describe different estimators.

Every number in this repository now comes from a customer-disjoint split, and
`tests/test_splits.py` fails the build if a row-level split is reintroduced.

Full findings: **[`docs/AUDIT.md`](docs/AUDIT.md)** — 26 defects in the
original notebook, each reproduced against the data before being recorded.

## Quick start

```bash
pip install -r requirements-dev.txt && pip install -e .

make train      # grouped CV, calibration, fairness audit, model card
make test       # 28 tests
make serve      # FastAPI on :8000, docs at /docs
```

Scoring an applicant:

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "Annual_Income": 19114.12, "Outstanding_Debt": 809.98,
  "Interest_Rate": 3, "Delay_from_due_date": 3,
  "Credit_Utilization_Ratio": 26.82, "Credit_History_Months": 265,
  "Credit_Mix": 2, "Occupation": "Scientist"
}'
```

```json
{
  "risk_band": "Good",
  "probabilities": {"Poor": 0.08, "Standard": 0.29, "Good": 0.63},
  "decision": "approve",
  "is_adverse": false,
  "reason_codes": [],
  "disclaimer": "..."
}
```

Declines carry `reason_codes` — the principal factors that drove the outcome,
in the shape 12 CFR 1002.9 requires of an adverse-action notice.

## What is in here

```
src/crediwise/
  config.py      schema, domain ranges, feature contract
  data.py        loading, cleaning, range validation
  features.py    ColumnTransformer -- everything fit inside CV folds
  splits.py      customer-grouped splitters + an overlap guard
  model.py       Random Forest + isotonic calibration
  evaluate.py    per-class PR/F1, AUC, Gini, KS, Brier, ECE
  fairness.py    disparate impact, equal opportunity, equalised odds
  explain.py     SHAP + ECOA-style reason codes
  policy.py      decision thresholds and cost matrix
  train.py       entry point
  api/           FastAPI service
docs/
  AUDIT.md            26 findings with reproduction evidence
  FAIRNESS.md         scope limits -- read before quoting any fairness number
  MODEL_CARD.md       generated from reports/metrics.json
  DATA_SHEET.md       provenance, defects, suitability
  FEATURE_CONTRACT.md what the API needs and what is excluded
notebooks/
  Credit_Score_original.ipynb   the audited original, kept for reference
```

## Design decisions worth knowing

**Splitting is grouped, always.** `splits.py` exposes no ungrouped splitter,
and `assert_no_group_overlap` raises rather than warns.

**Imputation lives inside the pipeline.** The original used `ffill()` on raw
row order, which pulled ~3,600 values across customer boundaries — a
customer's first missing month was filled from a different person. Median
imputation now fits on training folds only. Within-customer imputation scores
~0.7pp higher but cannot be reproduced for an applicant with no history, so
the serving-consistent choice wins.

**`Age` is excluded from the model.** It is ECOA-protected, and removing it
measures at **-0.02pp accuracy** — indistinguishable from noise, so it costs
nothing. It is retained only as a fairness audit dimension. This does *not* make the model age-neutral: the labels themselves
are age-correlated, which is why the audit measures outcomes rather than
inputs.

**`Credit_Mix` is declared, not hidden.** It is a bureau-assigned
credit-quality rating, so predicting a credit band from it is partly circular.
`make train-nobureau` quantifies the dependency at **-1.03pp accuracy and
-2.31 macro-F1**.

**Features were dropped on evidence, not convenience.** `Type_of_Loan`
multi-hot measured at -0.33pp macro-F1 during the audit, so it stays out.
`Credit_History_Age`, which the original dropped for being an awkward string,
parses to a month count and is worth +0.11pp accuracy under the shipped
pipeline — small, but free and in the right direction.

**Trees are deliberately shallow.** `min_samples_leaf=20` scores marginally
better than fully grown trees *and* keeps SHAP tractable — unpruned trees
average 11,332 leaves, pushing a 1,000-row explanation from ~5 minutes to
~47.

## Results

See **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)**, regenerated from
`reports/metrics.json` on every training run so it cannot drift from what the
model actually produced.

## License

MIT. The dataset is public synthetic data — see
[`docs/DATA_SHEET.md`](docs/DATA_SHEET.md).
