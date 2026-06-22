import pandas as pd
import numpy as np
from config import OBSERVATION_WINDOW_DAYS

def compute_rfm_features(obs_df, reference_date):
    """
    Computes RFM features and extensions from observation window transactions.
    reference_date is obs_end for the window.
    """
    df = obs_df.copy()

    rfm = df.groupby("customer_id").agg(
        recency=("invoicedate", lambda x: (reference_date - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary_total=("revenue", "sum"),
        monetary_avg=("revenue", "mean"),
        unique_products=("stockcode", "nunique")
    ).reset_index()

    # Rolling spend means: 30-day and 90-day
    df_30 = df[df["invoicedate"] >= reference_date - pd.Timedelta(days=30)]
    df_90 = df[df["invoicedate"] >= reference_date - pd.Timedelta(days=90)]

    spend_30 = df_30.groupby("customer_id")["revenue"].sum().rename("spend_30d")
    spend_90 = df_90.groupby("customer_id")["revenue"].sum().rename("spend_90d")

    rfm = rfm.merge(spend_30, on="customer_id", how="left")
    rfm = rfm.merge(spend_90, on="customer_id", how="left")
    rfm["spend_30d"] = rfm["spend_30d"].fillna(0)
    rfm["spend_90d"] = rfm["spend_90d"].fillna(0)

    # Time between purchases: mean and std
    purchase_intervals = df.groupby("customer_id")["invoicedate"].apply(
        lambda x: x.sort_values().diff().dt.days.dropna().agg(["mean", "std"])
    ).unstack().reset_index()
    purchase_intervals.columns = ["customer_id", "interpurchase_mean", "interpurchase_std"]
    purchase_intervals["interpurchase_mean"] = purchase_intervals["interpurchase_mean"].fillna(OBSERVATION_WINDOW_DAYS)
    purchase_intervals["interpurchase_std"] = purchase_intervals["interpurchase_std"].fillna(0)
    rfm = rfm.merge(purchase_intervals, on="customer_id", how="left")

    # Spend trend: slope of monthly spend
    df["month"] = df["invoicedate"].dt.to_period("M")
    monthly_spend = df.groupby(["customer_id", "month"])["revenue"].sum().reset_index()
    monthly_spend["month_idx"] = monthly_spend["month"].astype(str).astype("category").cat.codes

    def compute_slope(group):
        if len(group) < 2:
            return 0.0
        x = group["month_idx"].values
        y = group["revenue"].values
        slope = np.polyfit(x, y, 1)[0]
        return slope

    slopes = monthly_spend.groupby("customer_id").apply(compute_slope).rename("spend_trend")
    rfm = rfm.merge(slopes, on="customer_id", how="left")
    rfm["spend_trend"] = rfm["spend_trend"].fillna(0)

    # Category diversity: unique stockcodes as ratio of total transactions
    rfm["product_diversity"] = rfm["unique_products"] / rfm["frequency"]
    rfm["product_diversity"] = rfm["product_diversity"].fillna(0)

    # Seasonal drop-off flag: purchased in Q4 of previous year but not in current Q4
    if reference_date.month >= 10:
        current_q4_start = pd.Timestamp(year=reference_date.year, month=10, day=1)
        prev_q4_start = pd.Timestamp(year=reference_date.year - 1, month=10, day=1)
        prev_q4_end = pd.Timestamp(year=reference_date.year - 1, month=12, day=31)
    else:
        current_q4_start = pd.Timestamp(year=reference_date.year - 1, month=10, day=1)
        prev_q4_start = pd.Timestamp(year=reference_date.year - 2, month=10, day=1)
        prev_q4_end = pd.Timestamp(year=reference_date.year - 2, month=12, day=31)

    prev_q4_custs = set(df[(df["invoicedate"] >= prev_q4_start) & (df["invoicedate"] <= prev_q4_end)]["customer_id"])
    curr_q4_custs = set(df[(df["invoicedate"] >= current_q4_start) & (df["invoicedate"] <= reference_date)]["customer_id"])

    rfm["seasonal_dropoff"] = rfm["customer_id"].apply(
        lambda x: 1 if (x in prev_q4_custs and x not in curr_q4_custs) else 0
    )

    return rfm

def build_feature_matrix(windows):
    """
    Iterates over all windows, computes RFM features, attaches churn labels,
    and returns a single concatenated feature DataFrame.
    """
    all_features = []

    for i, window in enumerate(windows):
        obs_data = window["obs_data"]
        churn_labels = window["churn_labels"]
        reference_date = window["obs_end"]

        features = compute_rfm_features(obs_data, reference_date)
        features["churn"] = features["customer_id"].map(churn_labels)
        features["window_id"] = i
        features["obs_end"] = reference_date

        all_features.append(features)

    full_df = pd.concat(all_features, ignore_index=True)
    full_df["churn"] = full_df["churn"].astype(int)

    return full_df