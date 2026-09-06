import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, MODELS_DIR
from src.modeling.trainer import prepare_data, grouped_train_val_test_split
from src.evaluation.profit_optimizer import find_optimal_threshold, compute_avg_monthly_spend

feature_matrix_path = os.path.join(PROCESSED_DIR, "feature_matrix.pkl")
model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
calibrator_path = os.path.join(MODELS_DIR, "calibrator.pkl")
calibration_method_path = os.path.join(MODELS_DIR, "calibration_method.pkl")

for path in [feature_matrix_path, model_path, calibrator_path, calibration_method_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run scripts/run_pipeline.py first.")

feature_df = pd.read_pickle(feature_matrix_path)

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(calibrator_path, "rb") as f:
    calibrator = pickle.load(f)
with open(calibration_method_path, "rb") as f:
    calibration_method = pickle.load(f)

X, y, groups, feature_cols = prepare_data(feature_df)
X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
    grouped_train_val_test_split(X, y, groups)

raw_probs = model.predict_proba(X_test)[:, 1]
if calibration_method == "isotonic":
    calibrated_probs = calibrator.transform(raw_probs)
else:
    calibrated_probs = calibrator.predict_proba(np.array(raw_probs).reshape(-1, 1))[:, 1]

avg_monthly_spend = compute_avg_monthly_spend(
    feature_df.iloc[test_idx]["monetary_total"].reset_index(drop=True)
)

fine_thresholds = np.array([
    0.0001, 0.0005, 0.001, 0.002, 0.005,
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90
])
optimal_threshold, threshold_results = find_optimal_threshold(
    y_test.values, calibrated_probs, avg_monthly_spend, thresholds=fine_thresholds
)
threshold_results.insert(0, "requested_threshold", fine_thresholds)

print(f"Optimal threshold with floor 0.0001: {optimal_threshold}")
print()
print("Net profit across a log-spaced range down to 0.0001:")
print("(requested_threshold is the exact value tested; 'threshold' is rounded for display only)")
print(threshold_results.to_string(index=False))
