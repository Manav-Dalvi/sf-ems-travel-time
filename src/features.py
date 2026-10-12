"""Dispatch-time features for the travel-time model.

Every feature must be known when the unit is dispatched. Deliberately NOT used:
- month: one year of data with a time split, so test months (Oct-Dec) never appear in
  train; the model could only extrapolate from unrelated months.
- box: ~2,000 levels, many with few rows; lat/lon carry the same location information.
- neighborhoods_analysis_boundaries: derived from the location (redundant with lat/lon),
  and when DataSF assigns it is unconfirmed.
- call_type_group: forbidden until we confirm when it is assigned (project leakage rules).
- priority, final_priority: can be revised after dispatch (project leakage rules).
- call_final_disposition: only known when the call ends.
- any later timestamp (response_dttm, on_scene_dttm): after the prediction moment;
  on_scene_dttm is the target itself.
"""

import numpy as np
import pandas as pd

NUMERIC = ["lat", "lon", "als_unit", "unit_sequence_in_call_dispatch",
           "hour", "day_of_week", "workload_60m"]
CATEGORICAL = ["original_priority_cat", "station_area", "battalion", "unit_type"]
FEATURES = NUMERIC + CATEGORICAL

WORKLOAD_WINDOW = np.timedelta64(60, "m")


def workload_60m(received: pd.Series, raw: pd.DataFrame) -> np.ndarray:
    """Distinct calls in `raw` received in [t - 60 min, t) for each t in `received`.

    Uses the uncleaned file (all unit types) because that is the system load the
    dispatcher actually faces. Strictly earlier calls only, so a call never counts
    itself or anything received at the same instant or later.
    """
    per_call = raw.groupby("call_number")["received_dttm"]
    assert (per_call.nunique() == 1).all(), "workload assumes one received_dttm per call"
    times = np.sort(per_call.first().to_numpy())
    t = received.to_numpy()
    # side="left" at t excludes calls received exactly at t; at t-60m it includes the window start.
    return np.searchsorted(times, t, side="left") - np.searchsorted(times, t - WORKLOAD_WINDOW, side="left")


def check_workload_60m():
    ts = pd.Timestamp
    raw = pd.DataFrame({
        "call_number": ["Z", "A", "B", "B", "C", "D", "E"],
        "received_dttm": [ts("2025-01-01 09:59:59"),  # just outside the window
                          ts("2025-01-01 10:00:00"),  # window start: counted
                          ts("2025-01-01 10:30:00"),  # counted once despite two unit rows
                          ts("2025-01-01 10:30:00"),
                          ts("2025-01-01 11:00:00"),  # the call itself: not counted
                          ts("2025-01-01 11:00:00"),  # same timestamp, other call: not counted
                          ts("2025-01-01 11:10:00")],  # later call: not counted
    })
    got = workload_60m(pd.Series([ts("2025-01-01 11:00:00")]), raw)
    assert got.tolist() == [2], f"expected 2 (A, B), got {got.tolist()}"


def add_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["received_dttm"].dt.hour
    df["day_of_week"] = df["received_dttm"].dt.dayofweek
    df["als_unit"] = df["als_unit"].astype(int)
    df["workload_60m"] = workload_60m(df["received_dttm"], raw)
    return df


def make_X(df: pd.DataFrame, features=FEATURES, categories=None):
    """Feature matrix with categoricals fixed to the training levels.

    Pass `categories` from the training call when building X for other data, so codes
    line up and unseen levels (e.g. a held-out battalion's stations) become NaN.
    """
    X = df[features].copy()
    cat_cols = [c for c in CATEGORICAL if c in features]
    if categories is None:
        categories = {c: sorted(X[c].astype(str).unique()) for c in cat_cols}
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return X, categories


if __name__ == "__main__":
    check_workload_60m()
    print("check_workload_60m passed")
