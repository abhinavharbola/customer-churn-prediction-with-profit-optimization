import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COST_OF_OFFER, INTERVENTION_SUCCESS_RATE, MONTHS_REVENUE_SAVED

st.set_page_config(page_title="Churn Profit Optimizer", layout="wide")
st.title("Customer Churn Prediction with Profit Optimization")

MODEL_PATH = "models/xgb_model.pkl"
CALIBRATOR_PATH = "models/calibrator.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
CALIBRATION_METHOD_PATH = "models/calibration_method.pkl"
FEATURE_MATRIX_PATH = "data/processed/feature_matrix.pkl"

@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, CALIBRATOR_PATH, FEATURE_NAMES_PATH, CALIBRATION_METHOD_PATH]):
        st.error("Model artifacts not found. Run 'python run_pipeline.py' first.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    with open(CALIBRATION_METHOD_PATH, "rb") as f:
        calibration_method = pickle.load(f)

    def calibrate(scores):
        if calibration_method == "isotonic":
            return calibrator.transform(scores)
        else:
            return calibrator.predict_proba(np.array(scores).reshape(-1, 1))[:, 1]

    return model, calibrate, feature_names

@st.cache_data
def load_feature_matrix():
    if not os.path.exists(FEATURE_MATRIX_PATH):
        return None
    return pd.read_pickle(FEATURE_MATRIX_PATH)

def compute_expected_profit(prob, avg_monthly_spend):
    revenue_saved = avg_monthly_spend * MONTHS_REVENUE_SAVED
    expected_gain = prob * INTERVENTION_SUCCESS_RATE * revenue_saved
    return expected_gain - COST_OF_OFFER

model, calibrate, feature_names = load_artifacts()

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Analysis", "Model Info"])

with tab1:
    st.header("Customer Churn Assessment")

    input_mode = st.radio("Input mode", ["Manual Feature Entry", "Customer ID Lookup"])

    if input_mode == "Manual Feature Entry":
        st.subheader("RFM Features")

        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.slider("Recency (days)", 0, 365, 30)
            frequency = st.slider("Frequency", 1, 100, 5)
            monetary_total = st.number_input("Monetary Total (£)", 0.0, 50000.0, 500.0, step=50.0)
        with col2:
            monetary_avg = st.number_input("Avg Order Value (£)", 0.0, 5000.0, 100.0, step=10.0)
            unique_products = st.slider("Unique Products", 1, 200, 10)
            spend_30d = st.number_input("Spend 30 Days (£)", 0.0, 20000.0, 200.0, step=50.0)
        with col3:
            spend_90d = st.number_input("Spend 90 Days (£)", 0.0, 30000.0, 600.0, step=50.0)
            interpurchase_mean = st.number_input("Avg Days Between Purchases", 0.0, 365.0, 30.0)
            interpurchase_std = st.number_input("Std Days Between Purchases", 0.0, 200.0, 15.0)

        col4, col5 = st.columns(2)
        with col4:
            spend_trend = st.number_input("Spend Trend (slope)", -500.0, 500.0, 0.0, step=10.0)
        with col5:
            product_diversity = st.slider("Product Diversity", 0.0, 1.0, 0.5)
            seasonal_dropoff = st.selectbox("Seasonal Drop-off", [0, 1])

        input_values = np.array([[
            recency, frequency, monetary_total, monetary_avg, unique_products,
            spend_30d, spend_90d, interpurchase_mean, interpurchase_std,
            spend_trend, product_diversity, seasonal_dropoff
        ]])

        if frequency >= MONTHS_REVENUE_SAVED:
            avg_monthly_spend_for_profit = monetary_total / MONTHS_REVENUE_SAVED
        else:
            avg_monthly_spend_for_profit = monetary_total / max(frequency, 1)

    else:
        st.subheader("Lookup Customer by ID")

        feature_df = load_feature_matrix()
        if feature_df is None:
            st.error("Feature matrix not found. Run 'python run_pipeline.py' first to generate data/processed/feature_matrix.pkl")
            st.stop()

        available_ids = sorted(feature_df["customer_id"].unique())
        customer_id_input = st.selectbox("Select Customer ID", available_ids)

        customer_data = feature_df[feature_df["customer_id"] == customer_id_input]
        if customer_data.empty:
            st.error("Customer ID not found in feature matrix.")
            st.stop()

        latest_window = customer_data.sort_values("obs_end").iloc[-1]
        st.caption(f"Most recent observation window ending: {latest_window['obs_end'].strftime('%Y-%m-%d')}")

        input_values = np.array([[latest_window[col] for col in feature_names]])

        frequency_val = latest_window["frequency"]
        monetary_total_val = latest_window["monetary_total"]
        if frequency_val >= MONTHS_REVENUE_SAVED:
            avg_monthly_spend_for_profit = monetary_total_val / MONTHS_REVENUE_SAVED
        else:
            avg_monthly_spend_for_profit = monetary_total_val / max(frequency_val, 1)

    if st.button("Predict Churn Probability", type="primary"):
        if input_values is not None:
            input_df = pd.DataFrame(input_values, columns=feature_names)

            raw_prob = model.predict_proba(input_df)[:, 1][0]
            calibrated_prob = calibrate(np.array([raw_prob]))[0]
            if isinstance(calibrated_prob, np.ndarray):
                calibrated_prob = float(calibrated_prob)

            expected_profit = compute_expected_profit(calibrated_prob, avg_monthly_spend_for_profit)

            st.subheader("Results")
            prob_col, profit_col, action_col = st.columns(3)

            with prob_col:
                st.metric("Calibrated Churn Probability", f"{calibrated_prob:.3f}")
            with profit_col:
                st.metric("Expected Profit Delta", f"£{expected_profit:.2f}")
            with action_col:
                if expected_profit > 0:
                    st.success("INTERVENE")
                else:
                    st.error("DO NOT INTERVENE")

            st.subheader("Financial Breakdown")
            revenue_3m = avg_monthly_spend_for_profit * MONTHS_REVENUE_SAVED
            st.write(f"Revenue at stake (3 months): £{revenue_3m:.2f}")
            st.write(f"Probability of churn: {calibrated_prob:.3f}")
            st.write(f"Probability offer succeeds if churning: {INTERVENTION_SUCCESS_RATE}")
            st.write(f"Expected revenue saved: £{calibrated_prob * INTERVENTION_SUCCESS_RATE * revenue_3m:.2f}")
            st.write(f"Cost of intervention: £{COST_OF_OFFER:.2f}")
            st.write(f"Expected profit: £{expected_profit:.2f}")

            st.subheader("SHAP Explanation")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                fig, ax = plt.subplots()
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=input_values[0],
                        feature_names=feature_names
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

with tab2:
    st.header("Profit Optimization Analysis")

    COMPARISON_PATH = "data/processed/profit_comparison.csv"
    THRESHOLD_PATH = "data/processed/threshold_analysis.csv"

    if os.path.exists(COMPARISON_PATH):
        comparison_df = pd.read_csv(COMPARISON_PATH)
        st.subheader("Baseline Comparison")
        st.table(comparison_df)

        st.subheader("Insight")
        default_profit = float(comparison_df[comparison_df["Strategy"] == "Default Threshold (0.5)"]["Net Campaign Profit"].str.replace("£", "").str.replace(",", "").values[0])
        optimal_profit = float(comparison_df[comparison_df["Strategy"] == "Profit-Optimized"]["Net Campaign Profit"].str.replace("£", "").str.replace(",", "").values[0])
        lift_pct = ((optimal_profit - default_profit) / default_profit) * 100
        st.write(f"Profit-optimized threshold improves net profit by {lift_pct:.1f}% over the default 0.5 threshold.")

        if os.path.exists(THRESHOLD_PATH):
            threshold_df = pd.read_csv(THRESHOLD_PATH)
            st.subheader("Threshold Sweep")
            st.line_chart(threshold_df.set_index("threshold")["net_profit"])
            st.caption("Net profit at each threshold. The peak indicates the optimal decision boundary.")
    else:
        st.info("Profit comparison data not found. Run 'python run_pipeline.py' first.")

with tab3:
    st.header("Model Information")

    st.subheader("Architecture")
    st.write("""
    - **Model:** XGBoost classifier trained on natural class imbalance (no SMOTE, no class weighting).
    - **Calibration:** Isotonic regression mapping raw scores to calibrated probabilities.
    - **Features:** 12 RFM-based features computed over a 12-month observation window.
    - **Tuning:** 50 Optuna trials optimizing PR-AUC.
    """)

    st.subheader("Feature Descriptions")
    feature_descriptions = {
        "recency": "Days since last purchase in observation window.",
        "frequency": "Total number of unique invoices in observation window.",
        "monetary_total": "Total revenue generated in observation window.",
        "monetary_avg": "Average revenue per order.",
        "unique_products": "Number of distinct products purchased.",
        "spend_30d": "Total spend in the last 30 days of the observation window.",
        "spend_90d": "Total spend in the last 90 days of the observation window.",
        "interpurchase_mean": "Average days between consecutive purchases.",
        "interpurchase_std": "Standard deviation of days between purchases.",
        "spend_trend": "Slope of monthly spend over the observation window.",
        "product_diversity": "Ratio of unique products to total orders.",
        "seasonal_dropoff": "Purchased in Q4 of previous year but not current Q4.",
    }
    for feat, desc in feature_descriptions.items():
        st.write(f"- **{feat}:** {desc}")

    st.subheader("Configurable Parameters")
    st.write(f"""
    - Intervention cost: £{COST_OF_OFFER}
    - Intervention success rate: {INTERVENTION_SUCCESS_RATE}
    - Revenue saved horizon: {MONTHS_REVENUE_SAVED} months
    """)
    st.caption("Modify these in config.py to adapt to different business assumptions.")