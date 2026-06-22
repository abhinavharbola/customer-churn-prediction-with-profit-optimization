import numpy as np
import pandas as pd
from config import COST_OF_OFFER, INTERVENTION_SUCCESS_RATE, MONTHS_REVENUE_SAVED

def compute_expected_profit(prob, avg_monthly_spend):
    """
    Expected profit delta for a single customer.
    E[Delta Profit] = p * (gamma * V) - C
    where V = avg_monthly_spend * MONTHS_REVENUE_SAVED
    """
    revenue_saved = avg_monthly_spend * MONTHS_REVENUE_SAVED
    expected_gain = prob * INTERVENTION_SUCCESS_RATE * revenue_saved
    return expected_gain - COST_OF_OFFER

def find_optimal_threshold(y_true, y_prob, avg_monthly_spend, thresholds=np.arange(0.1, 0.91, 0.01)):
    """
    Searches threshold space to find the one maximizing total net profit.
    Returns optimal threshold and detailed results per threshold.
    """
    results = []

    for t in thresholds:
        predictions = (y_prob >= t).astype(int)
        true_positives = (predictions == 1) & (y_true == 1)
        false_positives = (predictions == 1) & (y_true == 0)

        tp_customers = true_positives.sum()
        fp_customers = false_positives.sum()
        total_interventions = tp_customers + fp_customers

        total_revenue_saved = 0.0
        if tp_customers > 0:
            tp_indices = np.where(true_positives)[0]
            tp_spends = avg_monthly_spend.iloc[tp_indices]
            total_revenue_saved = (INTERVENTION_SUCCESS_RATE * tp_spends * MONTHS_REVENUE_SAVED).sum()

        total_cost = total_interventions * COST_OF_OFFER
        net_profit = total_revenue_saved - total_cost

        results.append({
            "threshold": round(t, 2),
            "total_interventions": total_interventions,
            "true_positives": tp_customers,
            "false_positives": fp_customers,
            "revenue_saved": total_revenue_saved,
            "total_cost": total_cost,
            "net_profit": net_profit
        })

    results_df = pd.DataFrame(results)
    optimal_idx = results_df["net_profit"].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, "threshold"]

    return optimal_threshold, results_df

def evaluate_random_baseline(y_true, y_prob, avg_monthly_spend, fraction):
    np.random.seed(42)
    n = len(y_true)
    n_target = int(n * fraction)
    random_indices = np.random.choice(n, size=n_target, replace=False)

    predictions = np.zeros(n)
    predictions[random_indices] = 1

    true_positives = (predictions == 1) & (y_true == 1)
    false_positives = (predictions == 1) & (y_true == 0)

    tp_customers = true_positives.sum()
    fp_customers = false_positives.sum()
    total_interventions = tp_customers + fp_customers

    total_revenue_saved = 0.0
    if tp_customers > 0:
        tp_indices = np.where(true_positives)[0]
        tp_spends = avg_monthly_spend.iloc[tp_indices]
        total_revenue_saved = (INTERVENTION_SUCCESS_RATE * tp_spends * MONTHS_REVENUE_SAVED).sum()

    total_cost = total_interventions * COST_OF_OFFER
    net_profit = total_revenue_saved - total_cost

    return {
        "total_interventions": total_interventions,
        "true_positives": tp_customers,
        "false_positives": fp_customers,
        "revenue_saved": total_revenue_saved,
        "total_cost": total_cost,
        "net_profit": net_profit
    }

def evaluate_default_baseline(y_true, y_prob, avg_monthly_spend, threshold=0.5):
    predictions = (y_prob >= threshold).astype(int)

    true_positives = (predictions == 1) & (y_true == 1)
    false_positives = (predictions == 1) & (y_true == 0)

    tp_customers = true_positives.sum()
    fp_customers = false_positives.sum()
    total_interventions = tp_customers + fp_customers

    total_revenue_saved = 0.0
    if tp_customers > 0:
        tp_indices = np.where(true_positives)[0]
        tp_spends = avg_monthly_spend.iloc[tp_indices]
        total_revenue_saved = (INTERVENTION_SUCCESS_RATE * tp_spends * MONTHS_REVENUE_SAVED).sum()

    total_cost = total_interventions * COST_OF_OFFER
    net_profit = total_revenue_saved - total_cost

    return {
        "total_interventions": total_interventions,
        "true_positives": tp_customers,
        "false_positives": fp_customers,
        "revenue_saved": total_revenue_saved,
        "total_cost": total_cost,
        "net_profit": net_profit
    }