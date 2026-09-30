# Fairness scope and method

## Read this first: what this audit cannot do

ECOA and Regulation B name seven protected bases. **This dataset contains one
of them.**

| Protected basis | Present? |
|---|---|
| Race / colour | **Absent** |
| Religion | **Absent** |
| National origin | **Absent** |
| Sex | **Absent** |
| Marital status | **Absent** |
| Receipt of public assistance | **Absent** |
| Age | Present (8.5% of values corrupt) |

Worse, only **5 rows** fall in the 62-and-over bracket that ECOA specifically
protects, so the classic age-discrimination test is statistically empty here.

**Consequence:** nothing in this repository clears the model of race or sex
discrimination. A fairness report that measured only what happened to be
available, and presented it as a clean bill of health, would be exactly the
box-ticking exercise the audit warned about. What follows is bounded
deliberately.

## What is measured

Three slicing dimensions, all excluded from the model's inputs:

| Dimension | Status | Why |
|---|---|---|
| `age_band` | ECOA-protected | The one real protected basis available |
| `Occupation` | Proxy | Correlates with socioeconomic status |
| `income_quintile` | Proxy | Direct socioeconomic stratification |

Per subgroup, `crediwise.fairness` reports:

- **Selection rate** — share receiving the favourable outcome.
- **Disparate impact ratio** — lowest selection rate over highest. The
  four-fifths rule flags anything below 0.80.
- **Equal opportunity gap** — spread in true-positive rate among applicants
  who genuinely belong in the favourable band.
- **Equalised odds gap** — the larger of the TPR and FPR spreads.
- **Calibration gap** — spread in expected calibration error, i.e. whether a
  predicted 70% means the same thing for every group.

Subgroups below 100 rows are excluded rather than reported with unusable
error bars.

## Why removing `Age` is not enough

`Age` is not a model input. Dropping it was free — adding it back changes
accuracy by **+0.02pp**, inside run-to-run noise — so removing an
ECOA-protected attribute from the feature set costs nothing here.

But that does **not** make the model age-neutral, because the labels
themselves are age-correlated:

| Age band | n | Share labelled "Good" | Share labelled "Poor" |
|---|---|---|---|
| 18-25 | 21,652 | 16.2% | 30.7% |
| 26-35 | 28,319 | 15.9% | 31.1% |
| 36-45 | 27,621 | 16.3% | 30.5% |
| **46-55** | 13,559 | **32.9%** | **12.6%** |
| 56+ | 367 | 29.7% | 12.8% |

A better-than-2x gap in outcome rates is baked into the ground truth. A model
fit to these labels will reproduce it through correlated features whether or
not `Age` is present.

This is the central reason the audit measures **outcomes rather than inputs**.
"We removed the protected attribute" is not a fairness claim; it is a claim
about the feature list.

## Interpreting a failure

A disparate impact ratio below 0.80 is a flag, not a verdict. Under US lending
law a disparity can be lawful where it follows from a legitimate business
necessity with no less-discriminatory alternative. The correct response is
investigation — is the driver a genuine risk signal or a proxy? — not
automatically reweighting until the ratio passes.

Given the label skew above, expect the age-band test to show real disparity.
That finding is about the dataset as much as the model, and the model card
says so.

## Reproducing

```bash
make train                    # writes reports/metrics.json
python -c "import json; print(json.load(open('reports/metrics.json'))['fairness'])"
```
