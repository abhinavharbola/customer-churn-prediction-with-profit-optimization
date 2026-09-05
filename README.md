# Customer Churn Prediction with Profit Optimization

Predicting customer churn for non-contractual e-commerce using RFM features, calibrated XGBoost probabilities, and a cost-sensitive decision threshold that maximizes net campaign profit.

## Problem Statement

Standard classification metrics assume false positives and false negatives carry equal cost. In churn intervention, offering a retention discount to a customer who would have stayed anyway burns budget. Failing to catch a churning customer loses revenue. Maximizing ROC-AUC or F1 ignores this asymmetry.

This project replaces default 0.5 thresholding with an expected-value framework: the optimal threshold is the one that maximizes net profit after accounting for intervention cost, retention offer success rate, and the revenue at stake. Every customer receives an "INTERVENE" or "DO NOT INTERVENE" recommendation based strictly on whether the expected financial gain exceeds the intervention cost.

## Bug Fixes in This Revision

A full code audit (not just a prose pass) found and fixed five real bugs beyond the validation leak in [Key Results](#key-results). Each is covered by a regression test.

- **Cancellation netting corrupted unrelated line items** (`src/data/cleaner.py`). A cancellation was matched to a specific `(invoice, customer_id, stockcode)`, but the quantity correction was applied indexed by invoice number alone. Since one invoice can carry several stockcodes, the cancelled quantity was subtracted from every line on that invoice, not just the matched one. Fixed by indexing the correction on `(invoice, stockcode)`. Covered by `tests/test_cleaner.py`.
- **That same code path referenced a column that doesn't exist.** The merge that matches cancellations to transactions never produces a `quantity_cancel` column (only one side of the join has a `quantity` column, so pandas never suffixes it). On real data with any matched cancellation, this raised a `KeyError` and the pipeline never actually completed a run with the netting logic in place. Fixed alongside the point above.
- **The 3-month revenue horizon silently collapsed to the full 12-month total.** The dashboard computed `avg_monthly_spend = monetary_total / MONTHS_REVENUE_SAVED`, then `revenue_saved = avg_monthly_spend * MONTHS_REVENUE_SAVED`. The division and multiplication by the same constant cancel out exactly, so "3 months of revenue at stake" was actually each customer's entire 12-month spend, a 4x overstatement for any customer with 3 or more orders. The training pipeline used yet a third, different figure (`monetary_avg`, mean revenue per line item) for the same concept, so the threshold chosen during training wasn't even calibrated against what the dashboard used at serving time. Fixed with one shared `compute_avg_monthly_spend()` in `src/evaluation/profit_optimizer.py`, used identically by `scripts/run_pipeline.py` and `app/app.py`.
- **`seasonal_dropoff` was silently always 0 for about 75% of windows.** It compared calendar-Q4 date ranges, but those ranges frequently fall outside the 365-day observation window the feature is computed on. Whenever the window's reference date fell before October, the comparison period was mathematically guaranteed to be missing from the data. Redefined as a relative 90-day/180-day comparison that always fits inside the observation window regardless of calendar month. Covered by `tests/test_rfm_engineer.py`.
- **Manual feature entry built its model input by position, not by name.** It happened to match the model's actual feature order today, but nothing enforced that; a reorder in `rfm_engineer.py` would silently mislabel every value with no error. Now built as a name-keyed dict and reindexed against the model's own `feature_names`, matching the (already-safe) customer-lookup path, and fails loudly if a feature is missing instead of mislabeling it.
- **Batch Export's on-screen message didn't match its logic.** It told users scoring would use "the optimal threshold from training," but the code actually used per-customer expected value, which is the intentional, documented design (see Key Design Decisions). The message was inaccurate, not the logic. Fixed the message.
- **The cancellation fix above had two further edge cases that were still wrong on a second pass.** When two line items shared the exact same `(invoice, stockcode)` (duplicate lines, which occur in this dataset), the correction subtracted the cancelled quantity from every duplicate independently instead of netting once across them, over-cancelling. Deeper still, the join itself was the root cause: when the transactions side had duplicate `(invoice, customer_id, stockcode)` keys, a single cancellation record matched multiple transaction rows and got summed as if it were multiple separate cancellations, inflating the netted amount before any row-level logic even ran. Fixed by deduplicating the join key before matching, then netting sequentially across any remaining duplicate rows within a matched group. Covered by `test_duplicate_line_items_are_netted_not_double_cancelled` and `test_two_line_cancellation_against_same_product_sums_correctly`.
- **That same fix was also impractically slow at real scale.** The corrected logic ran a Python-level function over every `(invoice, stockcode)` group in the dataset, not just the ones with a matched cancellation. On a 300K-row synthetic test this didn't finish in a reasonable time; the real dataset is roughly 3-4x that size. Fixed by splitting the affected (matched) rows from the unaffected majority up front, running the row-by-row netting logic only on the small affected subset, and passing the rest through untouched. A 1M-row benchmark with 10,000 cancellations now completes in about 47 seconds.
- **The validation set was reused for tuning, calibration, and final reporting all at once.** Optuna's hyperparameter search directly optimized PR-AUC on the same set that was then used to fit the isotonic calibrator, compute the reported PR-AUC and Brier score, and select the profit-optimal threshold. Since the model's hyperparameters were chosen specifically to score well on that set, reporting performance on it again is optimistic by construction, the classic "tuning on the test set" problem, separate from the customer-overlap leak already described above. Fixed with a proper three-way customer-grouped split: `train` fits the model, `val` is Optuna's tuning target only, and `test` is held out completely from tuning and touched exactly once, for calibration and final reporting. Covered by `tests/test_grouped_split.py`, which now verifies pairwise disjointness across all three sets, not just two.
- **The formula panel in Model Info didn't nest correctly.** A styled `<div>` was opened in one `st.markdown()` call and closed in a separate call with `st.latex()` in between. Streamlit renders each call as an isolated block, so HTML can't be split across calls like that; the div would auto-close empty and the formula would render outside it, unstyled. Fixed by rendering the formula as a single self-contained HTML block instead of relying on `st.latex()`.
- **A Streamlit API version mismatch.** `width='stretch'` was used in some dataframe calls while `use_container_width=True` was used elsewhere for the same purpose. `width='stretch'` is a very recent addition and doesn't exist on the `streamlit>=1.25.0` floor this project pins in `requirements.txt`; it would raise a `TypeError` on that version. Standardized on `use_container_width=True`, which is supported across the whole pinned range.

## Architecture

- **Sliding temporal windows** prevent data leakage at the transaction level. A 12-month observation window, 90-day prediction window, and 30-day slide interval generate multiple training examples per customer across different seasonal periods.
- **Customer-grouped train/validation/test split** prevents leakage at the customer level. `GroupShuffleSplit`, keyed on `customer_id`, guarantees that no customer's windows cross a split boundary (`src/modeling/trainer.py`). This is a three-way split, not two: `train` fits the model, `val` is Optuna's tuning target, and `test` is held out completely from tuning and used only once, for calibration and final reporting. An earlier version of this pipeline split randomly across the concatenated feature matrix, which let the same customer appear in both sets through different windows; see [Bug Fixes in This Revision](#bug-fixes-in-this-revision) for what changed, including why a two-way split wasn't enough on its own.
- **Unweighted XGBoost** trained on the natural class imbalance preserves the true base churn rate. No SMOTE, no scale_pos_weight.
- **Isotonic regression** calibrates raw model scores into true probabilities, a strict requirement for valid expected-value calculations.
- **Profit optimizer** sweeps thresholds from 0.10 to 0.90, computes expected net profit at each step, and selects the argmax.
- **Batch export** generates a downloadable intervention list for campaign tools, with per-customer financial justification.

## Dataset

[Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1,067,371 raw transactional records from a UK-based online retailer spanning December 2009 to December 2011, combined across both sheets in the source `.xlsx`.

**Source:** UCI Machine Learning Repository. Direct download in `.xlsx` format containing two sheets, `Year 2009-2010` and `Year 2010-2011`. `src/data/cleaner.py` loads and concatenates both (`load_raw_data`).

The pipeline drops rows with a missing customer ID, nets cancellations against their matching invoice and stockcode, removes non-positive quantities and prices, and only then computes revenue and RFM features. That cleaning step changes the row count substantially from the raw 1,067,371. Run `python scripts/run_pipeline.py`; it prints the exact post-cleaning count as `Cleaned transactions: <n>`. That figure is intentionally not hardcoded here since it depends on the actual data file in `data/raw/`.

## Project Structure

```
churn-profit-opt/
├── config.py                    # All constants, paths, financial parameters
├── requirements.txt
├── .gitignore
├── scripts/
│   └── run_pipeline.py          # End-to-end training and evaluation script
├── src/
│   ├── data/
│   │   ├── cleaner.py           # Missing ID removal, invoice+stockcode cancellation netting
│   │   └── temporal.py          # Sliding window generator
│   ├── features/
│   │   └── rfm_engineer.py      # RFM + extensions computed per window
│   ├── modeling/
│   │   ├── trainer.py           # XGBoost with Optuna tuning, customer-grouped train/val/test split
│   │   └── calibrator.py        # Platt scaling / Isotonic regression
│   ├── evaluation/
│   │   ├── metrics.py           # PR-AUC, Brier score
│   │   ├── profit_optimizer.py  # Expected value maximization and baselines
│   │   └── explainability.py    # SHAP TreeExplainer wrapper used by the dashboard
├── app/
│   └── app.py                   # Streamlit interactive dashboard
├── tests/
│   ├── test_temporal.py
│   ├── test_rfm_engineer.py
│   ├── test_profit_optimizer.py
│   ├── test_grouped_split.py
│   └── test_cleaner.py
├── data/
│   ├── raw/                     # Place online_retail_II.xlsx here
│   └── processed/               # Generated feature matrices and results
├── models/                      # Serialized model, calibrator, feature names
└── assets/                      # Dashboard assets for README
```

## Setup

**1. Clone and create environment:**
```bash
git clone <repo-url>
cd churn-profit-opt
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**2. Download the dataset:**
Download `online_retail_II.xlsx` from the [UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in `data/raw/`.

**3. Run the pipeline:**
```bash
python scripts/run_pipeline.py
```
This executes cleaning, window generation, feature engineering, hyperparameter tuning (50 Optuna trials against a held-out validation split), the customer-grouped train/validation/test split, calibration and final reporting on a test split Optuna never saw, and profit optimization. Serialized model artifacts are saved to `models/`. Processed data and comparison results are saved to `data/processed/`.

**4. Run the tests:**
```bash
pytest tests/ -v
```

**5. Launch the dashboard:**
```bash
streamlit run app/app.py
```

## Key Results

The previous revision of this README reported PR-AUC 0.909, Brier score 0.135, an optimal threshold of 0.76, and a profit table built from a validation split that was random across the concatenated sliding-window feature matrix. That split let the same customer appear in both train and validation through different windows, which inflates every downstream number since the model could partly memorize customers it was evaluated on.

The split has been replaced with a customer-grouped, three-way split (`grouped_train_val_test_split` in `src/modeling/trainer.py`): train fits the model, validation is Optuna's tuning target, and test is held out completely from tuning and used only once, for calibration and final reporting. This closes two separate issues, not one: the original customer-overlap leak described above, and a second, independent problem found on a later audit pass, where the same validation set used for tuning was also being used to calibrate and report final metrics ("tuning on the test set"), which is optimistic regardless of the customer-overlap question. Both are verified by `tests/test_grouped_split.py`, which checks pairwise disjointness across all three sets and that every row lands in exactly one of them. That fix is in code and covered by tests, but this environment does not have network access to the UCI dataset, so the pipeline could not actually be re-run against the real file to regenerate PR-AUC, Brier score, the optimal threshold, or the profit comparison table. The old numbers are removed rather than left in place, since they were produced by the leaky split and would misrepresent the corrected pipeline.

To regenerate this table, run `python scripts/run_pipeline.py` end to end and copy the printed PR-AUC, Brier score, optimal threshold, and `data/processed/profit_comparison.csv` into this section. Expect PR-AUC and net profit to come in lower than the previous figures: the corrected split removes two separate sources of inflation, it does not add information.

## Dashboard Features

Four tabs provide a complete analytical and operational interface:

**Single Prediction:**
- Manual RFM feature entry with sliders or customer ID lookup from the processed feature matrix.
- Four-stat summary row: calibrated churn probability, expected profit, INTERVENE/DO NOT INTERVENE recommendation, and revenue at stake.
- Side-by-side financial breakdown and SHAP waterfall plot (`src/evaluation/explainability.py`, rendered via `shap.waterfall_plot`).
- Financial breakdown shows all components of the expected value calculation explicitly.

**Batch Analysis:**
- Profit comparison table across random, default threshold, and profit-optimized strategies.
- Profit lift percentage relative to the default threshold baseline.
- Threshold sweep chart visualizing net profit across the full threshold range, with the optimum marked.

**Model Info:**
- Three-column layout: architecture overview, feature descriptions, and configurable parameters.
- Profit formula displayed with LaTeX rendering.

**Batch Export:**
- One-click scoring of all customers using their latest observation window.
- Summary stats: total customers, intervene count, do-not-intervene count, total expected profit.
- Top 50 intervention candidates displayed by expected profit.
- Download the full intervention list as CSV for campaign tool import.
- Download the complete scored dataset for CRM integration or further analysis.

## Expected Value Framework

The profit calculation for each customer:

```
E[Profit] = P(churn) * (intervention_success_rate * avg_monthly_spend * 3 months) - intervention_cost
```

`avg_monthly_spend` is each customer's `monetary_total` (their revenue across the full 12-month observation window) divided by the number of months that window spans, computed once by `compute_avg_monthly_spend()` in `src/evaluation/profit_optimizer.py` and used identically during training and at serving time. It is not `monetary_total / MONTHS_REVENUE_SAVED`, which would cancel out against the `* MONTHS_REVENUE_SAVED` in this formula and silently turn "3 months of revenue" into the full 12 months; see [Bug Fixes in This Revision](#bug-fixes-in-this-revision).

A customer is targeted only when the expected profit is positive. The INTERVENE tag is not based on churn probability alone, it requires the expected financial gain to exceed the intervention cost. A customer with 30% churn probability but low monthly spend will be flagged DO NOT INTERVENE because the expected return is negative. This is the core differentiation from standard threshold-based classification.

## Configurable Parameters

All constants are centralized in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `OBSERVATION_WINDOW_DAYS` | 365 | Length of observation window |
| `PREDICTION_WINDOW_DAYS` | 90 | Churn lookahead period |
| `SLIDE_INTERVAL_DAYS` | 30 | Step size for sliding windows |
| `COST_OF_OFFER` | 10.0 | Cost per retention intervention (£) |
| `INTERVENTION_SUCCESS_RATE` | 0.15 | Fraction of churners who accept the offer |
| `MONTHS_REVENUE_SAVED` | 3 | Revenue horizon if customer is retained |
| `OPTUNA_TRIALS` | 50 | Number of hyperparameter search trials |
| `CALIBRATION_METHOD` | isotonic | isotonic or platt |
| `VALIDATION_SIZE` | 0.2 | Fraction of customers used for Optuna's tuning objective |
| `TEST_SIZE` | 0.2 | Fraction of customers held out completely from tuning, used only for calibration and final reporting |

Modify these to adapt the system to different business assumptions without changing pipeline code.

## Key Design Decisions

- **No SMOTE or class weighting:** Preserves true base churn rate for valid probability outputs. Post-hoc calibration corrects any score distortion from imbalance.
- **Isotonic over Platt scaling:** Non-parametric calibration handles the non-logistic score distortion typical of XGBoost on imbalanced data.
- **90-day churn window:** Balances false positive reduction against timely intervention. 30 days is too noisy; 180 days is too late.
- **Sliding windows over single split:** Prevents seasonal overfitting. E-commerce has strong Q4 effects that a single snapshot cannot capture.
- **Customer-grouped validation split:** A random row-level split lets a customer's other windows leak into validation. Grouping by `customer_id` removes that leak at the cost of a slightly smaller effective validation set.
- **PR-AUC over ROC-AUC:** ROC-AUC inflates performance on imbalanced data by rewarding correct ranking of abundant negatives.
- **Per-customer expected value over aggregate threshold:** The INTERVENE tag uses individual revenue and probability, not a population-level cutoff. This captures heterogeneity in customer value that a single threshold misses.

## Testing

Unit tests live in `tests/` and run offline against small synthetic fixtures, no dataset download required:

| File | Covers |
|---|---|
| `test_temporal.py` | Sliding window generation: windows are only produced when the data spans enough time, no observation-window transaction falls outside `[obs_start, obs_end)`, churn labels match prediction-window purchase presence. |
| `test_rfm_engineer.py` | RFM aggregation per window: recency, frequency, monetary total/avg, unique product counts, and `seasonal_dropoff` computed correctly and consistently across every calendar month. |
| `test_profit_optimizer.py` | Threshold sweep correctness, argmax selection, the expected-profit formula, and `compute_avg_monthly_spend` scaling correctly instead of collapsing to the full total. |
| `test_grouped_split.py` | The three-way split: no customer appears in more than one of train/val/test, every row is assigned to exactly one split, split sizes roughly match the configured fractions, and identifier columns are excluded from the feature set. |
| `test_cleaner.py` | Cancellation netting reduces only the matched `(invoice, stockcode)`, leaves unrelated line items and other invoices untouched, fully-cancelled lines are dropped, duplicate line items are netted correctly instead of double-cancelled, and multiple genuine cancellations against the same product sum correctly. |

33 tests total. Run them with:

```bash
pytest tests/ -v
```

## Limitations

- Cancellation matching assumes a credit note's invoice number is the original sale's invoice number with a `C` prefix. Orphaned refunds that don't follow this convention, or that reference an invoice outside the current cleaning batch, are left unmatched and simply pass through as separate negative-quantity rows removed by the `quantity > 0` filter, understating netted revenue rather than corrupting it.
- The three-way split can't stratify by churn label either (same `GroupShuffleSplit` limitation as before), now across three sets instead of two, which shrinks each further. With a few thousand customers this is usually minor, but it has not been measured against the real dataset in this revision.
- The Key Results numbers are pending a real re-run of the corrected pipeline; see that section for why.
- `monetary_avg` is the mean revenue per transaction line item, not per order/invoice. A customer with many low-value line items per order and one with few high-value line items can have the same `monetary_avg` despite very different order sizes. The dashboard labels this explicitly to avoid confusion with true average order value.
- The intervention success rate is a configurable constant, not a learned parameter. In production, this would come from A/B testing.
- Cold-start customers with no transaction history cannot be scored.
- The single optimal threshold reported in Batch Analysis is computed globally and used only for the baseline comparison table; it does not drive individual scoring decisions, which use per-customer expected value instead (see Key Design Decisions). Segment-specific thresholds per spend tier would still capture heterogeneity that a single global threshold misses in that comparison.
- The batch export uses the latest observation window per customer. Customers without a recent window are excluded.
