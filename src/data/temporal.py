import pandas as pd
import numpy as np
from datetime import timedelta
from config import OBSERVATION_WINDOW_DAYS, PREDICTION_WINDOW_DAYS, SLIDE_INTERVAL_DAYS, CHURN_THRESHOLD_DAYS

def generate_windows(df, observation_days=OBSERVATION_WINDOW_DAYS,
                     prediction_days=PREDICTION_WINDOW_DAYS,
                     slide_days=SLIDE_INTERVAL_DAYS):
    """
    Generates sliding window splits from transactional data.
    Each window has an observation period and a prediction period.
    Churn label is 1 if no purchase occurs in the prediction window.
    Returns a list of dictionaries with train/val dataframes.
    """
    df = df.copy()
    date_min = df["invoicedate"].min()
    date_max = df["invoicedate"].max()

    window_start = date_min
    windows = []

    while window_start + timedelta(days=observation_days + prediction_days) <= date_max:
        obs_start = window_start
        obs_end = obs_start + timedelta(days=observation_days)
        pred_end = obs_end + timedelta(days=prediction_days)

        # Observation period transactions
        obs_mask = (df["invoicedate"] >= obs_start) & (df["invoicedate"] < obs_end)
        obs_df = df[obs_mask].copy()

        # Prediction period transactions
        pred_mask = (df["invoicedate"] >= obs_end) & (df["invoicedate"] < pred_end)
        pred_df = df[pred_mask].copy()

        # Customers active in observation window
        obs_customers = set(obs_df["customer_id"].unique())

        # Customers who made a purchase in prediction window
        pred_customers = set(pred_df["customer_id"].unique())

        # Churn label: 0 if purchased in prediction window, 1 otherwise
        churn_labels = {cust_id: (0 if cust_id in pred_customers else 1) for cust_id in obs_customers}

        windows.append({
            "obs_start": obs_start,
            "obs_end": obs_end,
            "pred_end": pred_end,
            "obs_data": obs_df,
            "churn_labels": churn_labels
        })

        window_start += timedelta(days=slide_days)

    return windows