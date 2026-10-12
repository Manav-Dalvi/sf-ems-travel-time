"""80% prediction intervals: LightGBM quantile models, conformalized on valid (CQR)."""

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src import metrics
from src.features import FEATURES, make_X
from src.model import BASE_PARAMS, MAX_ITER, TARGET

COVERAGE = 0.8
ALPHAS = (0.1, 0.9)


def fit_quantile(train, valid, params, alpha):
    """Quantile model fit on train; iterations = minimum of the valid pinball-loss curve
    (early stopping with unlimited patience, capped at MAX_ITER)."""
    X, cats = make_X(train, FEATURES)
    Xv = make_X(valid, FEATURES, cats)[0]
    model = LGBMRegressor(**{**BASE_PARAMS, "objective": "quantile", "alpha": alpha},
                          **params, n_estimators=MAX_ITER)
    model.fit(X, train[TARGET], eval_set=[(Xv, valid[TARGET])], eval_metric="quantile")
    best = int(np.argmin(model.evals_result_["valid_0"]["quantile"])) + 1
    return model, cats, best


def predict_interval(models, df):
    """Raw [lo, hi] from the two quantile models, with crossed pairs put back in order."""
    (m_lo, c_lo, n_lo), (m_hi, c_hi, n_hi) = models
    lo = m_lo.predict(make_X(df, FEATURES, c_lo)[0], num_iteration=n_lo)
    hi = m_hi.predict(make_X(df, FEATURES, c_hi)[0], num_iteration=n_hi)
    crossed = int((lo > hi).sum())
    return (pd.Series(np.minimum(lo, hi), index=df.index),
            pd.Series(np.maximum(lo, hi), index=df.index), crossed)


def cqr_margin(y_cal, lo_cal, hi_cal, coverage=COVERAGE):
    """Conformalized quantile regression margin q from a calibration set.

    WHY: valid was not used to fit the quantile models' trees, so its conformity scores
    behave like fresh draws, and [lo - q, hi + q] has finite-sample marginal coverage
    >= `coverage` IF calibration and test rows are exchangeable. A time split only
    approximates that (test is a later season), so coverage on test is checked, not
    assumed. Caveat: valid did choose each quantile model's iteration count (one number
    per model), a mild reuse that weakens the guarantee slightly.
    """
    s = np.maximum(np.asarray(lo_cal) - np.asarray(y_cal), np.asarray(y_cal) - np.asarray(hi_cal))
    n = len(s)
    level = math.ceil((n + 1) * coverage) / n
    # method="higher" takes an actual order statistic, as the guarantee requires.
    q = float(np.quantile(s, min(level, 1.0), method="higher"))
    return q, level, n


def interval_compare(df, intervals: dict, segments=(None, "unit_type", "priority_group")):
    """Coverage % and mean width per interval method, overall and by segment."""
    rows = []
    for seg in segments:
        groups = [("ALL", df.index)] if seg is None else \
            [(f"{seg}={k}", idx) for k, idx in df.groupby(seg).groups.items()]
        for label, idx in groups:
            row = {"segment": label, "n": len(idx)}
            for name, (lo, hi) in intervals.items():
                rep = metrics.interval_report(df.loc[idx, TARGET], lo[idx], hi[idx])
                row[f"{name}_cov_%"] = round(100 * rep["coverage"], 1)
                row[f"{name}_width_s"] = round(rep["mean_width_s"], 1)
            rows.append(row)
    return pd.DataFrame(rows).set_index("segment")
