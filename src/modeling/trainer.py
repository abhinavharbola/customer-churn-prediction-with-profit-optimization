import xgboost as xgb
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score
from config import RANDOM_SEED, OPTUNA_TRIALS, VALIDATION_SIZE, TEST_SIZE


def prepare_data(feature_df):
    feature_cols = [c for c in feature_df.columns if c not in
                    ("customer_id", "churn", "window_id", "obs_end")]
    X = feature_df[feature_cols].copy()
    y = feature_df["churn"].copy()
    groups = feature_df["customer_id"].copy()
    return X, y, groups, feature_cols


def grouped_train_val_test_split(X, y, groups, test_size=TEST_SIZE,
                                  val_size=VALIDATION_SIZE, random_state=RANDOM_SEED):
    holdout_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(holdout_splitter.split(X, y, groups=groups))

    trainval_groups = groups.iloc[trainval_idx].reset_index(drop=True)
    relative_val_size = val_size / (1 - test_size)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx_rel, val_idx_rel = next(val_splitter.split(
        X.iloc[trainval_idx], y.iloc[trainval_idx], groups=trainval_groups
    ))

    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    train_customers = set(groups.iloc[train_idx])
    val_customers = set(groups.iloc[val_idx])
    test_customers = set(groups.iloc[test_idx])
    assert train_customers.isdisjoint(val_customers), "customer overlap between train and validation split"
    assert train_customers.isdisjoint(test_customers), "customer overlap between train and test split"
    assert val_customers.isdisjoint(test_customers), "customer overlap between validation and test split"

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_pred)


def train_model(X, y, groups, feature_cols):
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = \
        grouped_train_val_test_split(X, y, groups)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_params.update({
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model, study, feature_cols, X_test, y_test, test_idx
