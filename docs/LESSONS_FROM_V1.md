# Lessons from version 1

The first version of this project was a single notebook that sorted people into
Poor / Standard / Good credit-score brackets, using a synthetic dataset widely
shared on Kaggle. It reported **79.7% accuracy** with a Random Forest.

Auditing it found 26 problems. Five of them shaped how this version was built.

## 1. The same customer was in the training and the test data

The dataset had 8 monthly rows for each of 12,500 customers. Splitting the rows
at random put every customer's months on both sides, and since much of each
customer's record barely changed from month to month, the model could
*recognise* people instead of predicting anything.

Splitting by customer instead dropped accuracy from **79.7% to about 70%**.

**In this version:** one row per customer, duplicate records removed so no
record can appear on both sides, and the test set put aside before any other
work and scored once.

## 2. Some steps silently did nothing

The outlier filter saved its result to a variable that was never used, so ages
of −500 and 8,698 went straight into training. A feature-selection check was
computed, printed, and ignored, and its printed numbers were mathematically
impossible.

**In this version:** every step is code in `src/crediwise/` with tests, and the
notebook calls that code rather than doing its own processing.

## 3. Accuracy alone hid the weak spots

With imbalanced classes, accuracy looked fine while the smallest class did
poorly. Always guessing the most common answer already scored 53%.

**In this version:** accuracy isn't used as a headline at all (here, "nobody
defaults" is 78% accurate). The model is judged on ranking (AUC), on what its
decisions do to a loan book, and on whether its probabilities can be trusted,
always against a simple benchmark.

## 4. Personal data was committed, twice over

The dataset file held 12,501 social-security-number-shaped values and 10,139
names. The notebook's saved output also displayed one customer's name and SSN.
Both were purged from the repository's entire history.

**In this version:** the data is anonymised by its publisher, and the model
deliberately takes no personal attributes at all; the web service rejects them
if sent.

## 5. The data couldn't support real-world claims

No amount of better code could fix the dataset: it was synthetic, nobody knew
how its labels had been assigned, and it predicted a rating rather than
something that actually happened.

**In this version:** real customers, and a real outcome. Did they miss next
month's payment?

---

The original notebook is kept, unmodified, at
[`notebooks/Credit_Score_original.ipynb`](../notebooks/Credit_Score_original.ipynb).
It's there to read, not run: it needs the old dataset, which is no longer in the
repository, and library versions that are no longer current.

The full 26-point audit, with the evidence for each finding, is in the
repository history at commit `eba480f` (`docs/AUDIT.md`):

```bash
git show eba480f:docs/AUDIT.md
```
