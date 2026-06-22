import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from config import CALIBRATION_METHOD

def calibrate_probabilities(model, X_val, y_val, method=CALIBRATION_METHOD):
    """
    Calibrates model probability outputs using isotonic regression or Platt scaling.
    Returns a calibration function that maps raw scores to calibrated probabilities.
    """
    raw_scores = model.predict_proba(X_val)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_scores, y_val)

        def calibration_func(scores):
            return calibrator.transform(scores)

    elif method == "platt":
        calibrator = LogisticRegression()
        calibrator.fit(raw_scores.reshape(-1, 1), y_val)

        def calibration_func(scores):
            return calibrator.predict_proba(np.array(scores).reshape(-1, 1))[:, 1]

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return calibration_func, calibrator