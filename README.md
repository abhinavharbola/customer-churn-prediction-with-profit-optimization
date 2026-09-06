# Customer Churn Prediction with Profit Optimization

Given a customer transaction history, this pipeline predicts churn probability, calibrates it into a true probability, and turns that into a per-customer INTERVENE / DO NOT INTERVENE call based on whether the expected financial gain of a retention offer exceeds its cost, never a raw probability cutoff alone.

Built entirely on free, open infrastructure: the public [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset, open-source libraries (XGBoost, Optuna, scikit-learn, Streamlit), no paid APIs, no GPU required.

## Preview

<p align="center">
  <img src="assets/manual_entry.png" width="720" alt="Streamlit dashboard showing manual RFM feature entry with sliders and number inputs, a four-stat result row, and a SHAP waterfall explanation">
  <br>
  <sub><em>Single Prediction, manual feature entry: churn probability, expected profit, the INTERVENE/DO NOT INTERVENE call, and a per-feature SHAP explanation for it.</em></sub>
</p>

## What this is

Given the raw transaction file, the pipeline:

1. Cleans it, drops rows with no customer ID, nets cancellations against the specific line item they credit (not the whole invoice), keeps only positive quantities and prices.
2. Builds sliding observation/prediction windows per customer, so the same customer contributes multiple labeled examples across different points in time, not one static snapshot.
3. Engineers 12 RFM-based features per customer per window, and labels churn by whether the customer bought again in the following 90 days.
4. Tunes XGBoost with Optuna against a customer-grouped validation split, then calibrates and reports on a customer-grouped test split that tuning never saw.
5. Sweeps decision thresholds to find the one that maximizes net campaign profit, not the one that maximizes accuracy.
6. Serves all of this through a Streamlit dashboard: single-customer lookup, batch scoring, and a downloadable intervention list, every recommendation traceable back to the expected-value formula behind it.

## Architecture

```mermaid
flowchart TD
    raw[online_retail_II.xlsx] --> clean[Cleaner\nmissing IDs dropped, cancellations netted per line item]
    clean --> windows[Sliding windows\n365d observation / 90d prediction / 30d slide]
    windows --> features[RFM feature engineer\n12 features per customer per window]
    features --> split[Customer-grouped split\ntrain / val / test, no customer crosses a boundary]
    split -->|train| tune[XGBoost + Optuna\n50 trials, tuned against val only]
    split -->|val| tune
    tune --> calibrate[Isotonic calibration\nfit on test]
    split -->|test, untouched by tuning| calibrate
    calibrate --> metrics[PR-AUC / Brier\non test]
    calibrate --> profit[Profit optimizer\nthreshold sweep + baselines]
    profit --> artifacts[(models/ + data/processed/)]
    artifacts --> dashboard[Streamlit dashboard\nSingle Prediction, Batch Analysis, Model Info, Batch Export]
```

`scripts/check_threshold_floor.py` runs a reduced path through this same architecture: it reloads the saved model and calibrator, reproduces the identical train/val/test split (deterministic given the fixed `RANDOM_SEED`), and re-sweeps thresholds alone, skipping cleaning, windowing, feature engineering, and tuning entirely. That's only possible because the split is reproducible by construction, not incidental.

## Expected Value Framework

```
E[Profit] = P(churn) * (intervention_success_rate * avg_monthly_spend * 3 months) - intervention_cost
```

`avg_monthly_spend` is each customer's `monetary_total` (revenue across the full 12-month observation window) divided by the number of months that window spans, computed once by `compute_avg_monthly_spend()` in `src/evaluation/profit_optimizer.py` and used identically during training and at serving time, never two different figures for the same concept in different places.

A customer is targeted only when expected profit is positive. A customer with a high churn probability but low monthly spend can still be correctly flagged DO NOT INTERVENE, because the expected return doesn't clear the intervention cost. This is the core difference from thresholding on probability alone.

## Models

| Role | Model | Library | Notes |
|---|---|---|---|
| Churn classifier | XGBoost (`XGBClassifier`) | `xgboost` | Trained on the natural class imbalance. No SMOTE, no `scale_pos_weight`; post-hoc calibration corrects score distortion instead. |
| Hyperparameter search | TPE sampler (Optuna's default) | `optuna` | 50 trials, objective is validation-set PR-AUC only. Never sees the test set, by construction (see Guardrails). |
| Calibration | Isotonic regression | `scikit-learn` | Default; Platt scaling (`LogisticRegression`) is available via `CALIBRATION_METHOD` in `config.py`. Fit on the test set, converts raw scores into the true probabilities the profit formula requires. |
| Explainability | SHAP `TreeExplainer` | `shap` | Per-customer waterfall plot in the dashboard only. Never touches training, tuning, or thresholding. |

The same isolation principle that keeps an eval judge structurally separate from what it's judging applies here between the validation and test splits: the set that picks the hyperparameters is never the set the final numbers get reported on.

## Guardrails

- **Split integrity is asserted, not assumed.** `grouped_train_val_test_split()` (`src/modeling/trainer.py`) asserts pairwise customer disjointness across train/val/test on every call; a regression here fails loudly in `tests/test_grouped_split.py`, not silently in a reported number.
- **The test set is touched exactly once.** Used only for calibration and final metrics, never for tuning. This project reports real numbers only when it can trace them to an actual run against real data, not a placeholder carried over from a leaky split (see Evaluation).
- **Cancellation netting is scoped, not global.** A cancellation only decrements the specific `(invoice, stockcode)` it matches, verified against duplicate-line-item and multi-cancellation edge cases in `tests/test_cleaner.py`, not just the common case.
- **Graceful degradation on missing artifacts.** Every dashboard tab checks for its required model/data files before using them and shows a clear "run the pipeline first" message instead of a raw traceback.
- **Graceful degradation on a zero or negative baseline.** The profit-lift calculation in Batch Analysis falls back to an absolute currency delta instead of dividing by zero or reporting a nonsensical percentage over a negative base.

## Artifacts

`scripts/run_pipeline.py` persists everything the dashboard needs to `models/` (`xgb_model.pkl`, `calibrator.pkl`, `feature_names.pkl`, `calibration_method.pkl`) and `data/processed/` (`feature_matrix.pkl`, `profit_comparison.csv`, `threshold_analysis.csv`). `app/app.py` only ever reads these; it never retrains.

Re-running the pipeline overwrites all of the above in place. There's no versioning and no run history kept, if you want to compare two configurations, save a copy of `data/processed/` and `models/` before re-running with different `config.py` values.

## Data Integrity

Cancellation matching assumes a credit note's invoice number is the original sale's invoice number with a `C` prefix, since that's the convention this specific dataset follows. This is a mitigation for a known messy-data pattern, not a guarantee: a refund that doesn't follow the convention is simply left unmatched and passes through as a separate negative-quantity row removed by the `quantity > 0` filter. It understates netted revenue in that case rather than corrupting an unrelated row, which was the actual bug this replaced (see the bug log at the bottom of this file).

## Dataset

[Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii): 1,067,371 raw transactional records from a UK-based online retailer spanning December 2009 to December 2011, combined across both sheets (`Year 2009-2010`, `Year 2010-2011`) in the source `.xlsx`. `src/data/cleaner.py` loads and concatenates both (`load_raw_data`).

The post-cleaning row count isn't hardcoded here since it depends on the actual file in `data/raw/`; `scripts/run_pipeline.py` prints it as `Cleaned transactions: <n>`.

## Project Structure

```
churn-profit-opt/
├── .streamlit/
│   └── config.toml               # Explicit theme, so the app doesn't depend on OS/browser dark-mode
├── config.py                    # All constants, paths, financial parameters
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── run_pipeline.py          # End-to-end training and evaluation script
│   └── check_threshold_floor.py # Reuses saved artifacts to re-sweep thresholds without retraining
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
│   ├── test_temporal.py         # sliding window boundaries + churn label correctness
│   ├── test_rfm_engineer.py     # RFM aggregation math, seasonal_dropoff across all calendar months
│   ├── test_profit_optimizer.py # threshold sweep, argmax, compute_avg_monthly_spend scaling
│   ├── test_grouped_split.py    # pairwise train/val/test disjointness, every row assigned once
│   └── test_cleaner.py          # cancellation netting, duplicate line items, multi-cancellation sums
├── data/
│   ├── raw/                     # Place online_retail_II.xlsx here
│   └── processed/               # Generated feature matrices and results
├── models/                      # Serialized model, calibrator, feature names
└── assets/                      # Dashboard assets for README
```

## Getting started

1. **Get the dataset:** download `online_retail_II.xlsx` from the [UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in `data/raw/`.

2. **Install:**
   ```bash
   git clone <repo-url>
   cd churn-profit-opt
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Review `config.py` (optional).** You don't need to touch it for a first run, but every financial and windowing assumption lives there:

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
   | `TEST_SIZE` | 0.2 | Fraction of customers held out completely from tuning |

## Running it

```bash
python scripts/run_pipeline.py     # cleans, tunes XGBoost, calibrates, saves artifacts to models/ and data/processed/
pytest tests/ -v                   # 33 tests, offline, small synthetic fixtures, no dataset needed
streamlit run app/app.py           # dashboard; needs the artifacts from the pipeline run above
```

The dashboard has four tabs. **Single Prediction** takes manual RFM input or a customer ID lookup and returns a four-stat result row (churn probability, expected profit, the INTERVENE/DO NOT INTERVENE call, revenue at stake) alongside a SHAP explanation. **Batch Analysis** compares net profit across random, default-threshold, and profit-optimized strategies, with the full threshold sweep charted. **Model Info** documents the architecture, every feature, and the profit formula in one place. **Batch Export** scores every customer at once and produces a downloadable intervention list, each row carrying its own financial justification, plus the full scored dataset for CRM or campaign-tool import.

## Evaluation

Evaluation here means two different things, and this project doesn't blur them: whether the code is correct (unit tests), and whether the model is actually good (held-out metrics). Conflating them is how leakage bugs like the one below hide for a while.

- **Unit tests** (`tests/`, `pytest tests/ -v`) run offline against small synthetic fixtures and check code correctness, not model quality: sliding-window boundaries, RFM aggregation math, the profit formula's arithmetic, cancellation-netting edge cases, and split disjointness. 33 tests total, see the file-by-file breakdown in [Project Structure](#project-structure).
- **Model quality** is PR-AUC, Brier score, and net profit, computed once on the held-out test split described in Models and Guardrails above, using `scripts/run_pipeline.py`. PR-AUC rather than ROC-AUC on purpose: ROC-AUC inflates performance on class-imbalanced data like this by rewarding correct ranking of the abundant negative class.
- **Those numbers, on the real dataset:**

  | Metric | Value |
  |---|---|
  | PR-AUC (held-out test) | 0.8233 |
  | Brier score (held-out test) | 0.1781 |
  | Optimal threshold | 0.02 |

  | Strategy | Total Interventions | True Positives | Wasted Spend (FP) | Net Campaign Profit |
  |---|---|---|---|---|
  | Random (20%) | 1,710 | 1,061 | 649 | £21,532 |
  | Default Threshold (0.5) | 6,400 | 4,724 | 1,676 | £60,063 |
  | Profit-Optimized (0.02) | 8,368 | 5,294 | 3,074 | £100,998 |

  A previous revision of this README reported PR-AUC 0.909 and a profit table from a split that let the same customer leak between train and validation, numbers that were real outputs but measured the wrong thing. The split is now fixed and verified by `tests/test_grouped_split.py`, and the table above is from an actual `scripts/run_pipeline.py` run against the real dataset, not a re-hash of the old, leaky figures.

  The optimal threshold of 0.02 was specifically checked against a wider, log-spaced sweep down to 0.0001 (`scripts/check_threshold_floor.py`) to rule out the same search-boundary artifact that caused the original 0.1 finding: net profit is flat from 0.0001 to 0.01 (isotonic calibration maps that whole range to the same set of customers), rises to its peak at 0.02, then declines monotonically from 0.08 onward. Profit is lower on both sides of 0.02, the signature of a genuine local maximum, not a boundary effect. The 8,368-intervention, £100,998 result at threshold 0.02 was independently reproduced twice, once from the full pipeline run and once from the standalone threshold check reusing the same saved model against the same deterministically-reconstructed test split, and the two agree exactly.

  Expect these exact figures to shift on a re-run with a different `RANDOM_SEED`, a different `data/raw/online_retail_II.xlsx` snapshot, or after any change to `config.py`'s financial assumptions; they are not fixed constants of the method.


## Known limitations

- Cancellation matching's `C`-prefix convention is a mitigation, not a guarantee; see Data Integrity above.
- The three-way split can't stratify by churn label (a `GroupShuffleSplit` limitation), and splitting three ways shrinks each set further than a two-way split would. With a few thousand customers this is usually minor but hasn't been measured against the real dataset.
- `monetary_avg` is mean revenue per transaction line item, not per order/invoice; the dashboard labels it explicitly to avoid implying true average order value.
- The intervention success rate is a configurable constant, not a learned parameter; in production this would come from A/B testing.
- Cold-start customers with no transaction history cannot be scored.
- The single global optimal threshold shown in Batch Analysis is used only for that baseline comparison table; it does not drive individual scoring decisions, which use per-customer expected value instead.
- Batch export uses each customer's latest observation window; customers without a recent window are excluded.