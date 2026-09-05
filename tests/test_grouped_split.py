import pandas as pd
import numpy as np
from src.modeling.trainer import prepare_data, grouped_train_val_test_split


def _synthetic_feature_df():
    rows = []
    for customer_id in range(150):
        for window_id in range(4):
            rows.append({
                "customer_id": customer_id,
                "window_id": window_id,
                "obs_end": pd.Timestamp("2010-01-01") + pd.Timedelta(days=30 * window_id),
                "recency": np.random.randint(0, 100),
                "frequency": np.random.randint(1, 10),
                "monetary_total": np.random.uniform(10, 500),
                "churn": np.random.randint(0, 2),
            })
    return pd.DataFrame(rows)


def test_no_customer_appears_in_more_than_one_split():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    _, _, _, _, _, _, train_idx, val_idx, test_idx = grouped_train_val_test_split(X, y, groups)

    train_customers = set(groups.iloc[train_idx])
    val_customers = set(groups.iloc[val_idx])
    test_customers = set(groups.iloc[test_idx])

    assert train_customers.isdisjoint(val_customers)
    assert train_customers.isdisjoint(test_customers)
    assert val_customers.isdisjoint(test_customers)


def test_all_windows_of_a_customer_stay_in_one_split():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    _, _, _, _, _, _, train_idx, val_idx, test_idx = grouped_train_val_test_split(X, y, groups)

    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    test_groups = set(groups.iloc[test_idx])

    for customer_id in groups.unique():
        memberships = sum([
            customer_id in train_groups,
            customer_id in val_groups,
            customer_id in test_groups,
        ])
        assert memberships == 1


def test_every_row_is_assigned_to_exactly_one_split():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    _, _, _, _, _, _, train_idx, val_idx, test_idx = grouped_train_val_test_split(X, y, groups)

    all_idx = sorted(list(train_idx) + list(val_idx) + list(test_idx))
    assert all_idx == list(range(len(X)))


def test_split_sizes_roughly_match_configured_fractions():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    _, _, _, _, _, _, train_idx, val_idx, test_idx = grouped_train_val_test_split(X, y, groups)

    n_customers = groups.nunique()
    train_frac = len(set(groups.iloc[train_idx])) / n_customers
    val_frac = len(set(groups.iloc[val_idx])) / n_customers
    test_frac = len(set(groups.iloc[test_idx])) / n_customers

    assert 0.45 < train_frac < 0.75
    assert 0.1 < val_frac < 0.35
    assert 0.1 < test_frac < 0.35


def test_feature_columns_exclude_identifiers():
    feature_df = _synthetic_feature_df()
    X, y, groups, feature_cols = prepare_data(feature_df)

    assert "customer_id" not in feature_cols
    assert "churn" not in feature_cols
    assert "window_id" not in feature_cols
    assert "obs_end" not in feature_cols
