import os
import pickle
import pandas as pd
import numpy as np
from config import PROCESSED_DIR, DEFAULT_THRESHOLD, RANDOM_TARGET_FRACTION, RANDOM_SEED, CALIBRATION_METHOD

from src.data.cleaner import run_cleaning
from src.data.temporal import generate_windows
from src.features.rfm_engineer import build_feature_matrix
from src.modeling.trainer import prepare_data, train_model
from src.modeling.calibrator import calibrate_probabilities
from src.evaluation.metrics import compute_metrics
from src.evaluation.profit_optimizer import (
    find_optimal_threshold,
    evaluate_random_baseline,
    evaluate_default_baseline
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=== 1. Cleaning raw data ===")
df_clean = run_cleaning()
print(f"Cleaned transactions: {len(df_clean)}")

print("=== 2. Generating temporal windows ===")
windows = generate_windows(df_clean)
print(f"Windows generated: {len(windows)}")

print("=== 3. Building feature matrix ===")
feature_df = build_feature_matrix(windows)
print(f"Feature matrix shape: {feature_df.shape}")
print(f"Churn rate: {feature_df['churn'].mean():.3f}")

feature_df.to_pickle(os.path.join(PROCESSED_DIR, "feature_matrix.pkl"))

print("=== 4. Training XGBoost with Optuna ===")
X, y, feature_cols = prepare_data(feature_df)
model, study, feature_cols, X_val, y_val = train_model(X, y, feature_cols)
print(f"Best trial PR-AUC: {study.best_value:.4f}")

print("=== 5. Calibrating probabilities ===")
calibration_func, calibrator = calibrate_probabilities(model, X_val, y_val)

raw_probs = model.predict_proba(X_val)[:, 1]
calibrated_probs = calibration_func(raw_probs)

print("=== 6. Evaluating metrics ===")
metrics = compute_metrics(y_val, calibrated_probs)
print(f"PR-AUC: {metrics['pr_auc']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")

print("=== 7. Profit optimization ===")
val_indices = y_val.index
avg_monthly_spend = feature_df.loc[val_indices, "monetary_avg"].copy()

optimal_threshold, threshold_results = find_optimal_threshold(
    y_val.values, calibrated_probs, avg_monthly_spend
)
print(f"Optimal threshold: {optimal_threshold}")

print("=== 8. Baseline comparison ===")
random_result = evaluate_random_baseline(
    y_val.values, calibrated_probs, avg_monthly_spend, RANDOM_TARGET_FRACTION
)
default_result = evaluate_default_baseline(
    y_val.values, calibrated_probs, avg_monthly_spend, DEFAULT_THRESHOLD
)
optimal_result = threshold_results[threshold_results["threshold"] == optimal_threshold].iloc[0]

comparison_df = pd.DataFrame({
    "Strategy": ["Random (20%)", "Default Threshold (0.5)", "Profit-Optimized"],
    "Total Interventions": [
        random_result["total_interventions"],
        default_result["total_interventions"],
        int(optimal_result["total_interventions"])
    ],
    "True Positives": [
        random_result["true_positives"],
        default_result["true_positives"],
        int(optimal_result["true_positives"])
    ],
    "Wasted Spend (FP)": [
        random_result["false_positives"],
        default_result["false_positives"],
        int(optimal_result["false_positives"])
    ],
    "Net Campaign Profit": [
        f"£{random_result['net_profit']:,.0f}",
        f"£{default_result['net_profit']:,.0f}",
        f"£{optimal_result['net_profit']:,.0f}"
    ]
})

print("\n=== Profit Comparison ===")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(os.path.join(PROCESSED_DIR, "profit_comparison.csv"), index=False)
threshold_results.to_csv(os.path.join(PROCESSED_DIR, "threshold_analysis.csv"), index=False)

print("=== 9. Saving artifacts ===")
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f)
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open("models/calibration_method.pkl", "wb") as f:
    pickle.dump(CALIBRATION_METHOD, f)

print("\nPipeline complete. Run 'streamlit run app/app.py' for dashboard.")