"""Point and interval error metrics, all in seconds."""

import numpy as np

WITHIN_S = 120


def report(y_true, y_pred) -> dict:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    abs_err = np.abs(err)
    return {
        "MAE": abs_err.mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MedAE": np.median(abs_err),
        "P90AE": np.quantile(abs_err, 0.9),
        f"within±{WITHIN_S}s": (abs_err <= WITHIN_S).mean(),
    }


def interval_report(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return {"coverage": ((y >= lo) & (y <= hi)).mean(), "mean_width_s": (hi - lo).mean()}
