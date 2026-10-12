"""Point and interval error metrics, all in seconds."""

import numpy as np
import pandas as pd

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


def paired_cluster_bootstrap(y, pred_a, pred_b, groups, n_boot=1000, seed=0) -> pd.DataFrame:
    """95% CI for MAE(a) - MAE(b) and RMSE(a) - RMSE(b), resampling whole groups.

    Rows of one call share conditions, so resampling rows would understate the
    uncertainty; whole calls are drawn with replacement instead. Both models are
    scored on the same resample (paired), so their shared noise cancels.
    """
    y, pred_a, pred_b = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    codes, uniques = pd.factorize(np.asarray(groups))
    n_g = len(uniques)

    def per_group(v):
        return np.bincount(codes, weights=v, minlength=n_g)

    n = per_group(np.ones_like(y))
    abs_a, abs_b = per_group(np.abs(pred_a - y)), per_group(np.abs(pred_b - y))
    sq_a, sq_b = per_group((pred_a - y) ** 2), per_group((pred_b - y) ** 2)

    # Elementwise sums rather than `@`: macOS Accelerate BLAS emits spurious FP warnings on matmul.
    def diffs(w):
        rows = (w * n).sum()
        return (((w * abs_a).sum() - (w * abs_b).sum()) / rows,
                np.sqrt((w * sq_a).sum() / rows) - np.sqrt((w * sq_b).sum() / rows))

    rng = np.random.default_rng(seed)
    boot = np.array([diffs(np.bincount(rng.integers(0, n_g, n_g), minlength=n_g))
                     for _ in range(n_boot)])
    point = diffs(np.ones(n_g))
    return pd.DataFrame({
        "estimate": point,
        "ci_2.5%": np.percentile(boot, 2.5, axis=0),
        "ci_97.5%": np.percentile(boot, 97.5, axis=0),
    }, index=["MAE(a) - MAE(b)", "RMSE(a) - RMSE(b)"])
