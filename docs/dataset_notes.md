# IEEE-CIS Fraud Detection Dataset — Engineering Notes

Source: https://www.kaggle.com/c/ieee-fraud-detection/data

These are the engineering gotchas that will break your model if you ignore them.
Read this before touching the feature pipeline.

---

## 1. TransactionDT is NOT a Unix timestamp

`TransactionDT` is seconds elapsed since a reference point, **not** a Unix timestamp.
The reference date (inferred from context) is approximately **2017-11-30**.

```python
# Wrong — will give you dates in 1970
pd.to_datetime(df["TransactionDT"], unit="s")

# Right
REFERENCE = pd.Timestamp("2017-11-30")
actual_time = REFERENCE + pd.to_timedelta(df["TransactionDT"], unit="s")
```

The dataset spans approximately 6 months (Dec 2017 – Jun 2018). This matters for
time-aware splitting — the split boundary is around TransactionDT = 10,000,000.

---

## 2. ~60% of transactions have no identity row

The `train_identity.csv` has 144,233 rows vs 590,540 in `train_transaction.csv`.
This means ~60% of transactions have no device/browser/OS information.

This is NOT a data quality problem. It reflects real-world behaviour:
- Mobile app transactions may not capture device info
- Some payment methods don't capture identity fields

**Consequence:** All identity-derived features must handle NaN as a valid,
informative state (not just as "missing data"). Use null indicator features.

---

## 3. The V-columns (V1–V339) are anonymised and correlated

339 anonymised features, grouped by Vesta's internal feature families.
They're highly correlated within families (r > 0.95 is common).

What we know from the Kaggle competition:
- V-columns are likely Vesta's proprietary transaction features
- Within-family columns often encode the same signal at different aggregation levels
- Dropping all but one per correlated cluster reduces 339 → ~100 without losing signal

**Our approach:** Correlation pruning — drop any V-column that has r > 0.95 with
another V-column. This is done in `build_features.py`.

---

## 4. Class imbalance: ~3.5% fraud rate

`isFraud` distribution: ~96.5% legitimate, ~3.5% fraud.

This breaks naive accuracy as a metric. A model that predicts "not fraud" for
every transaction gets 96.5% accuracy and is useless.

Our three-layer response:
1. **SMOTE** on training data — synthesise minority class samples
2. **class_weight="balanced"** in LightGBM — up-weight fraud samples
3. **Precision-Recall AUC** as primary metric — not ROC-AUC, not accuracy

Why PR-AUC and not ROC-AUC?
- ROC-AUC includes the true negative rate, which is trivially high (most transactions are legit)
- PR-AUC focuses on your ability to correctly identify fraud without too many false alarms
- A random classifier's PR-AUC baseline = class prior = 0.035 (not 0.5 like ROC-AUC)

---

## 5. High cardinality in card and device fields

| Column | Unique values | Notes |
|---|---|---|
| card1 | 13,553 | Likely a card/account identifier |
| card2 | 500 | |
| DeviceInfo | 1,786 | Raw device strings ("Samsung Galaxy S9") |
| id_30 | 99 | OS + version ("Windows 10") |
| id_31 | 139 | Browser + version ("chrome 69.0") |

One-hot encoding DeviceInfo creates 1,786 sparse columns. LightGBM handles this
poorly and logistic regression can't handle it at all.

**Our approach:**
- `TargetEncoder` for email domains and device fields (encode with mean fraud rate)
- `FrequencyEncoder` for card identifiers (encode with appearance count)

---

## 6. Missing values are informative, not random

Column null rates vary widely:
- `dist2`: ~93% null
- `id_*` columns: 40–70% null
- `D*` (timedelta) columns: 20–50% null

A null in `id_20` means "no identity row" — which may correlate with fraud.
A null in `dist2` may mean "no billing address distance" — also potentially informative.

**Our approach:** Add binary `_was_null` flags for key columns before imputation.
This preserves the "was this null?" signal even after the null is filled.

---

## 7. Time-based leakage in rolling features

Velocity features (spend in last 1h, 24h per card) must only use past transactions.
At training time, using the full dataset to compute rolling stats leaks future
information into the model.

For this portfolio project, we compute proxy velocity features (total card spend,
card transaction count) on the training set only, which approximates the real pattern.

In production, you would use a real-time feature store (Redis, Feast) to serve
pre-computed rolling windows at inference time. This is documented as a natural
next step in the architecture.

---

## 8. The M-columns (M1–M9) are match flags

`M1`–`M9` contain "T" (True), "F" (False), and NaN.
They represent match results (name match, address match, etc.) between the
purchase and billing records.

These are ordinal: map to {-1: NaN, 0: "F", 1: "T"}.

---

## Column naming conventions

| Prefix | What it is |
|---|---|
| `Transaction*` | Core transaction fields |
| `card1–card6` | Card/payment method attributes |
| `addr1, addr2` | Billing address zip code areas |
| `dist1, dist2` | Address distance features |
| `P_emaildomain` | Purchaser email domain |
| `R_emaildomain` | Recipient email domain |
| `C1–C14` | Counting features (Vesta internal) |
| `D1–D15` | Timedelta features (days since prior event) |
| `M1–M9` | Match/flag features |
| `V1–V339` | Vesta anonymised features |
| `id_01–id_38` | Identity table features |
| `DeviceType` | "desktop" or "mobile" |
| `DeviceInfo` | Raw device string |
