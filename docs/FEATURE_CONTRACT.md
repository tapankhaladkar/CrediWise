# Feature contract

What the scoring service requires, where each field comes from, and what is
deliberately excluded. `src/crediwise/api/schemas.py` enforces this.

## Inputs

### Applicant- or lender-supplied (17)

`Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`,
`Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `Delay_from_due_date`,
`Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`,
`Outstanding_Debt`, `Credit_Utilization_Ratio`, `Total_EMI_per_month`,
`Amount_invested_monthly`, `Monthly_Balance`, `Credit_History_Months`,
`Payment_of_Min_Amount`

Plus two nominal fields: `Occupation`, `Payment_Behaviour`.

All are optional. Missing values are imputed with the medians learned at
training time, so a partial record still scores.

### Bureau-supplied (1)

`Credit_Mix` — a bureau assessment of the applicant's mix of credit types
(0 = Bad, 1 = Standard, 2 = Good).

**This field is circular** (audit #15): it is itself a credit-quality
judgement, so predicting a credit band from it is partly predicting a rating
from a rating. It is the third most important feature in the model.

The dependency is quantified rather than hidden:

Measured with the shipped pipeline on the grouped holdout set:

| Variant | Accuracy | Macro-F1 |
|---|---|---|
| With `Credit_Mix` | 69.86% | 67.65 |
| **Without** (`--no-bureau`) | **68.83%** | **65.34** |
| Cost of dropping it | **-1.03pp** | **-2.31** |

Note the macro-F1 cost (-2.31) is more than twice the accuracy cost, because
the loss falls mainly on the minority classes.

If a deployment cannot obtain a bureau credit-mix rating before scoring, train
and serve the `--no-bureau` variant and accept that cost. Do not serve
the full model with `Credit_Mix` imputed to the median — that silently
substitutes the population average for the single most consequential input.

## Deliberate exclusions

| Field | Why excluded |
|---|---|
| `Name`, `SSN` | Direct identifiers, removed from the dataset entirely |
| `ID` | Row identifier, no predictive content |
| `Customer_ID` | Panel key — used to group splits, never a feature |
| `Age` | ECOA-protected. Removal measured at **-0.02pp**, within noise, so it is free. Retained only as a fairness audit dimension |
| `Month` | Artifact of the panel layout; the forest was assigning it 3.5% importance |
| `Type_of_Loan` | Multi-hot encoding measured at **-0.33pp macro-F1** during the audit; dropped on evidence |

## Output

```json
{
  "risk_band": "Standard",
  "probabilities": {"Poor": 0.21, "Standard": 0.58, "Good": 0.21},
  "decision": "refer",
  "is_adverse": false,
  "cost_optimal_band": "Standard",
  "reason_codes": [],
  "model_version": "0.2.0",
  "disclaimer": "..."
}
```

`reason_codes` is populated whenever `is_adverse` is true, giving the
principal factors that drove the outcome (12 CFR 1002.9). They describe what
moved *this model's* output and are not a legally reviewed adverse-action
notice.
