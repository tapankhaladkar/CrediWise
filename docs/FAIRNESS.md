# Fairness

## Why check outcomes, not just inputs

The model is never given sex, marital status, age or education. That alone
doesn't make it fair: other features can stand in for them. Credit limits,
for instance, tend to be lower for younger customers, so a model can treat age
groups differently without ever seeing age. So the audit checks **what the
model does to each group**, using the test set.

## What is measured

For every group with at least 100 customers in the test set:

| Measure | Question it answers |
|---|---|
| **Approval rate** | How often is this group approved? The lowest group's rate divided by the highest's is the *approval ratio*; the "four-fifths rule" used by US regulators flags anything under 0.8. |
| **Approval rate among customers who repaid** | Are *creditworthy* people treated alike, whatever their group? This is the most direct fairness question (equal opportunity). |
| **Predicted vs actual default rate** | Is the model's risk estimate accurate for this group, or too harsh or too lenient? |
| **Amplification** | Groups that genuinely repay at different rates will be approved at different rates. This compares the approval ratio with the ratio of actual repayment rates. 1 means approvals differ in proportion to real repayment; **below 1, the model widens the gap**; above 1, it narrows it. |

"Unknown" groups (undocumented codes) are shown but never used as the lowest
or highest group: they are missing-data buckets, not groups of people.

## Results

From [`MODEL_CARD.md`](MODEL_CARD.md), which is regenerated on every
training run. At the time of writing:

| Group | Approval ratio | Four-fifths rule | Amplification | Gap in approval among those who repaid |
|---|---|---|---|---|
| Sex | 0.941 | pass | 0.99, in line with repayment | 3.4 points |
| Marital status | 0.995 | pass | 1.04, slightly narrower | 2.0 points |
| Age band | 0.902 | pass | 0.95, slightly wider | **6.2 points** |
| Education | 0.846 | pass | 0.94, slightly wider | 5.2 points |

All four pass. The model mostly mirrors real differences in repayment, and
widens them slightly for age and education.

### The finding to watch: customers in their twenties

| Age | Actual default | Predicted default | Approved, among those who repaid |
|---|---|---|---|
| **21–29** | 21.7% | **23.6%** | **61%** |
| 30–39 | 20.8% | 20.6% | 68% |

The model overestimates risk for customers aged 21–29 by about two points, and
creditworthy customers in that group are approved noticeably less often than
those in their thirties: the largest equal-opportunity gap in the audit.

A likely cause, **not yet verified**: younger customers have lower credit
limits, so the same bill uses more of their limit, and both "credit limit" and
"share of limit in use" are among the model's top features. The next step would
be to compare the model's explanations for young customers who repaid with
those for older ones, and check whether limit-related features are doing the
damage.

Under US lending law a disparity can be lawful if it follows from a genuine
risk factor and no less discriminatory alternative exists. A flag like this
calls for investigation, not automatic reweighting.

## What this audit cannot tell you

- **Four of the seven protected bases aren't in the data:** race or colour,
  religion, national origin, and receipt of public assistance. Nothing here
  speaks to them.
- **Some groups are too small to assess** in the test set and are listed in the
  model card rather than dropped silently, e.g. customers aged 60–79 (67) and
  "other" marital status (57).
- **The data is from Taiwan.** US rules are used as a benchmark because they
  are concrete and widely understood, not because they govern this data.
- **One test set, one point in time.** Group sizes of a few thousand mean
  differences of a point or two are within ordinary variation.

## Reproducing

`make train` runs the audit and writes it to `reports/metrics.json`;
`docs/MODEL_CARD.md` and section 8 of the walkthrough notebook show it in full.
