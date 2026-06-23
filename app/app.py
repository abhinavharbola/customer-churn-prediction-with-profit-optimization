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
        calibrator_obj = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    with open(CALIBRATION_METHOD_PATH, "rb") as f:
        calibration_method = pickle.load(f)

    def calibrate(scores):
        if calibration_method == "isotonic":
            return calibrator_obj.transform(scores)
        else:
            return calibrator_obj.predict_proba(np.array(scores).reshape(-1, 1))[:, 1]

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

tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Analysis", "Model Info", "Batch Export"])

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

            st.divider()

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Churn Probability", f"{calibrated_prob:.1%}")
            with metric_col2:
                st.metric("Expected Profit", f"£{expected_profit:,.2f}")
            with metric_col3:
                if expected_profit > 0:
                    st.success("**INTERVENE**")
                else:
                    st.error("**DO NOT INTERVENE**")
            with metric_col4:
                revenue_3m = avg_monthly_spend_for_profit * MONTHS_REVENUE_SAVED
                st.metric("Revenue at Stake", f"£{revenue_3m:,.2f}")

            st.divider()

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.caption("FINANCIAL BREAKDOWN")
                breakdown_data = {
                    "Avg monthly spend": f"£{avg_monthly_spend_for_profit:,.2f}",
                    "Revenue at stake (3 months)": f"£{revenue_3m:,.2f}",
                    "Intervention success rate": f"{INTERVENTION_SUCCESS_RATE:.0%}",
                    "Expected revenue saved": f"£{calibrated_prob * INTERVENTION_SUCCESS_RATE * revenue_3m:,.2f}",
                    "Intervention cost": f"£{COST_OF_OFFER:,.2f}",
                    "Expected profit": f"£{expected_profit:,.2f}",
                }
                for label, value in breakdown_data.items():
                    st.write(f"{label}: {value}")

            with detail_col2:
                st.caption("SHAP EXPLANATION")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)

                    fig, ax = plt.subplots(figsize=(5, 3))
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

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.subheader("Architecture")
        st.write("""
        - **Model:** XGBoost
        - **Training:** Natural imbalance, no SMOTE, no weighting
        - **Calibration:** Isotonic regression
        - **Features:** 12 RFM-based features
        - **Window:** 12-month observation, 90-day prediction
        - **Tuning:** 50 Optuna trials optimizing PR-AUC
        """)

    with info_col2:
        st.subheader("Feature Descriptions")
        feature_descriptions = {
            "recency": "Days since last purchase",
            "frequency": "Unique invoices in window",
            "monetary_total": "Total revenue in window",
            "monetary_avg": "Avg revenue per order",
            "unique_products": "Distinct products purchased",
            "spend_30d": "Spend in last 30 days",
            "spend_90d": "Spend in last 90 days",
            "interpurchase_mean": "Avg days between purchases",
            "interpurchase_std": "Std days between purchases",
            "spend_trend": "Slope of monthly spend",
            "product_diversity": "Unique products / total orders",
            "seasonal_dropoff": "Q4 drop-off flag",
        }
        for feat, desc in feature_descriptions.items():
            st.write(f"- **{feat}:** {desc}")

    with info_col3:
        st.subheader("Configurable Parameters")
        st.write(f"""
        - Intervention cost: **£{COST_OF_OFFER}**
        - Success rate: **{INTERVENTION_SUCCESS_RATE:.0%}**
        - Revenue horizon: **{MONTHS_REVENUE_SAVED} months**
        """)

        st.subheader("Profit Formula")
        st.latex(r"E[\Delta Profit] = p_i \cdot (\gamma \cdot V_i) - C")
        st.caption("""
        - p_i: calibrated churn probability
        - γ: intervention success rate
        - V_i: avg monthly spend × 3 months
        - C: intervention cost (£10)
        """)

    st.caption("Modify parameters in config.py to adapt to different business assumptions.")

with tab4:
    st.header("Export Intervention List")

    feature_df = load_feature_matrix()
    if feature_df is None:
        st.error("Feature matrix not found. Run 'python run_pipeline.py' first to generate data/processed/feature_matrix.pkl")
        st.stop()

    THRESHOLD_PATH = "data/processed/threshold_analysis.csv"

    if not os.path.exists(THRESHOLD_PATH):
        st.error("Threshold analysis not found. Run 'python run_pipeline.py' first.")
        st.stop()

    threshold_df = pd.read_csv(THRESHOLD_PATH)
    optimal_threshold = threshold_df.loc[threshold_df["net_profit"].idxmax(), "threshold"]

    st.write(f"Using optimal threshold from training: **{optimal_threshold}**")

    if st.button("Score All Customers", type="primary"):
        with st.spinner("Scoring customers..."):

            latest_windows = feature_df.sort_values("obs_end").groupby("customer_id").last().reset_index()

            X_export = latest_windows[feature_names].copy()

            raw_probs = model.predict_proba(X_export)[:, 1]
            calibrated_probs = calibrate(raw_probs)
            if isinstance(calibrated_probs, np.ndarray):
                calibrated_probs = calibrated_probs.flatten()

            latest_windows["calibrated_churn_prob"] = calibrated_probs

            latest_windows["avg_monthly_spend"] = latest_windows.apply(
                lambda row: row["monetary_total"] / MONTHS_REVENUE_SAVED
                if row["frequency"] >= MONTHS_REVENUE_SAVED
                else row["monetary_total"] / max(row["frequency"], 1),
                axis=1
            )

            latest_windows["expected_profit"] = latest_windows.apply(
                lambda row: compute_expected_profit(row["calibrated_churn_prob"], row["avg_monthly_spend"]),
                axis=1
            )

            latest_windows["intervention_decision"] = latest_windows["expected_profit"].apply(
                lambda x: "INTERVENE" if x > 0 else "DO NOT INTERVENE"
            )

            latest_windows["revenue_at_stake"] = latest_windows["avg_monthly_spend"] * MONTHS_REVENUE_SAVED

            export_columns = [
                "customer_id", "calibrated_churn_prob", "avg_monthly_spend",
                "revenue_at_stake", "expected_profit", "intervention_decision",
                "recency", "frequency", "monetary_total", "obs_end"
            ]
            available_export_cols = [c for c in export_columns if c in latest_windows.columns]
            results_df = latest_windows[available_export_cols].copy()
            results_df = results_df.sort_values("expected_profit", ascending=False)

            intervene_df = results_df[results_df["intervention_decision"] == "INTERVENE"]
            do_not_df = results_df[results_df["intervention_decision"] == "DO NOT INTERVENE"]

            st.success(f"Scoring complete. {len(results_df)} customers evaluated.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(results_df))
            with col2:
                st.metric("Intervene", len(intervene_df))
            with col3:
                st.metric("Do Not Intervene", len(do_not_df))

            st.subheader("Intervention List (Top 50 by Expected Profit)")
            st.dataframe(
                intervene_df.head(50).style.format({
                    "calibrated_churn_prob": "{:.3f}",
                    "avg_monthly_spend": "£{:,.2f}",
                    "revenue_at_stake": "£{:,.2f}",
                    "expected_profit": "£{:,.2f}",
                    "monetary_total": "£{:,.2f}"
                }),
                width='stretch'
            )

            st.download_button(
                label="Download Full Intervention List (CSV)",
                data=intervene_df.to_csv(index=False),
                file_name="intervention_list.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download Full Results (CSV)",
                data=results_df.to_csv(index=False),
                file_name="all_customers_scored.csv",
                mime="text/csv"
            )