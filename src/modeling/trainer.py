import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from config import RANDOM_SEED, OPTUNA_TRIALS

def prepare_data(feature_df):
    feature_cols = [c for c in feature_df.columns if c not in
                    ("customer_id", "churn", "window_id", "obs_end")]
    X = feature_df[feature_cols].copy()
    y = feature_df["churn"].copy()
    return X, y, feature_cols

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

def train_model(X, y, feature_cols):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

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

    return model, study, feature_cols, X_val, y_val