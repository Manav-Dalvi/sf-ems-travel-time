"""Train/valid/test time split and spatial CV folds, both grouped by call_number."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Inclusive date ranges on received_dttm (whole days).
SPLIT_RANGES = {
    "train": ("2025-01-01", "2025-08-31"),
    "valid": ("2025-09-01", "2025-09-30"),
    "test": ("2025-10-01", "2025-12-31"),
}

MIN_BATTALION_ROWS = 500
MAX_LOBO_GROUPS = 12


def assign_time_split(df: pd.DataFrame) -> pd.Series:
    """Return a 'split' label per row, assigned per call_number from its received date."""
    # A call must have one received time, otherwise "which split" would be ambiguous.
    n_received = df.groupby("call_number")["received_dttm"].nunique()
    assert (n_received == 1).all(), f"{(n_received > 1).sum()} calls have >1 received_dttm"

    call_day = df.groupby("call_number")["received_dttm"].first().dt.normalize()
    call_split = pd.Series(pd.NA, index=call_day.index, dtype="string")
    for name, (start, end) in SPLIT_RANGES.items():
        call_split[call_day.between(pd.Timestamp(start), pd.Timestamp(end))] = name
    assert call_split.notna().all(), "some calls fall outside every split range"

    split = df["call_number"].map(call_split)
    assert (split.groupby(df["call_number"]).nunique() == 1).all(), "a call_number spans two splits"
    return split


def split_counts(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    counts = pd.DataFrame({"rows": split.value_counts(),
                           "calls": df.groupby(split)["call_number"].nunique()})
    return counts.reindex(list(SPLIT_RANGES))


def spatial_folds(train_df: pd.DataFrame, verbose=True) -> np.ndarray:
    """Fold id per row, leave-one-battalion-out, for testing generalisation to unseen areas."""
    rows_per_bat = train_df["battalion"].value_counts()
    small = rows_per_bat[rows_per_bat < MIN_BATTALION_ROWS].index
    # Tiny battalions would make noisy one-off folds; pool them instead.
    group = train_df["battalion"].where(~train_df["battalion"].isin(small), "other")
    n_groups = group.nunique()

    if verbose:
        print("train rows per battalion:")
        print(rows_per_bat.to_string())
        if len(small):
            print(f"merged into 'other' (< {MIN_BATTALION_ROWS} rows): {list(small)}")
        else:
            print(f"no battalion has < {MIN_BATTALION_ROWS} rows; nothing merged")

    if n_groups > MAX_LOBO_GROUPS:
        raise NotImplementedError(
            f"{n_groups} battalion groups > {MAX_LOBO_GROUPS}; fold count not yet decided")

    folds = np.empty(len(train_df), dtype=int)
    for fold, (_, test_idx) in enumerate(GroupKFold(n_splits=n_groups).split(train_df, groups=group)):
        folds[test_idx] = fold

    # Battalion is constant within a call in this data, so calls can't straddle folds; check anyway.
    assert (pd.Series(folds).groupby(train_df["call_number"].to_numpy()).nunique() == 1).all(), \
        "a call_number sits in more than one fold"
    if verbose:
        print(f"\nleave-one-battalion-out: {n_groups} folds")
        print(pd.crosstab(group.to_numpy(), folds, rownames=["battalion"], colnames=["fold"]).to_string())
    return folds
