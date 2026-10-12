"""Stage 1 cleaning: turn raw unit responses into the ambulance modelling dataset.

Rules are applied in a fixed order and every removal is counted, so the exclusion
table can be reported alongside any result.
"""

import pandas as pd

AMBULANCE_UNIT_TYPES = {"MEDIC", "PRIVATE"}

# 2025 US/Pacific DST transitions, as naive local wall-clock windows (+/- 30 min
# around the skipped / repeated hour). Assumes raw timestamps are naive local time.
DST_WINDOWS_2025 = [
    (pd.Timestamp("2025-03-09 01:30"), pd.Timestamp("2025-03-09 03:30")),  # spring forward
    (pd.Timestamp("2025-11-02 00:30"), pd.Timestamp("2025-11-02 02:30")),  # fall back
]


def travel_time_seconds(df: pd.DataFrame) -> pd.Series:
    return (df["on_scene_dttm"] - df["response_dttm"]).dt.total_seconds()


def overlaps_window(df: pd.DataFrame, start, end) -> pd.Series:
    """True where [received_dttm, on_scene_dttm] intersects [start, end]."""
    return (df["received_dttm"] <= end) & (df["on_scene_dttm"] >= start)


def overlaps_any_dst_window(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for start, end in DST_WINDOWS_2025:
        mask |= overlaps_window(df, start, end)
    return mask


def apply_rules(raw_df, unit_types, min_tt_s=None, max_tt_s=None):
    """Apply R1-R7 with R1 keeping `unit_types`. Returns (clean_df, exclusion_table)."""
    df = raw_df.copy()
    df["travel_time_s"] = travel_time_seconds(df)
    all_unit_types = sorted(raw_df["unit_type"].dropna().unique())

    rules = [
        # R1: the target is ambulance travel; engines/other units respond under different patterns.
        ("R1", f"keep unit_type in {sorted(unit_types)}",
         lambda d: d["unit_type"].isin(unit_types)),
        # R2: location and station/box are needed for any spatial feature; can't impute them honestly.
        ("R2", "drop missing lat, lon, station_area, or box",
         lambda d: d[["lat", "lon", "station_area", "box"]].notna().all(axis=1)),
        # R3: a unit cannot arrive at or before the moment it started moving.
        ("R3", "drop travel_time_s <= 0",
         lambda d: d["travel_time_s"] > 0),
        # R4: out-of-order timestamps mean at least one is wrong, so travel time is untrustworthy.
        ("R4", "drop dispatch < received or response < dispatch",
         lambda d: ~((d["dispatch_dttm"] < d["received_dttm"])
                     | (d["response_dttm"] < d["dispatch_dttm"]))),
        # R5: naive local timestamps across a DST shift can be off by +/- 1 hour.
        ("R5", "drop [received, on_scene] overlapping a 2025 DST window",
         lambda d: ~overlaps_any_dst_window(d)),
    ]
    # R6/R7: plausibility bounds; thresholds not yet chosen, so off by default.
    if min_tt_s is not None:
        rules.append(("R6", f"drop travel_time_s < {min_tt_s}",
                      lambda d: d["travel_time_s"] >= min_tt_s))
    if max_tt_s is not None:
        rules.append(("R7", f"drop travel_time_s > {max_tt_s}",
                      lambda d: d["travel_time_s"] <= max_tt_s))

    records = []
    for rule, description, keep_fn in rules:
        keep = keep_fn(df)
        removed_by_type = df.loc[~keep, "unit_type"].value_counts()
        records.append({
            "rule": rule,
            "description": description,
            "rows_before": len(df),
            "rows_removed": int((~keep).sum()),
            "rows_after": int(keep.sum()),
            **{f"removed_{t}": int(removed_by_type.get(t, 0)) for t in all_unit_types},
        })
        df = df[keep]

    # Keep original_priority's letter codes as-is (their meaning is unconfirmed);
    # "missing" is an explicit category rather than silently dropped.
    df["original_priority_cat"] = df["original_priority"].astype("string").fillna("missing").astype(str)
    return df.reset_index(drop=True), pd.DataFrame(records)


def build_ambulance_dataset(raw_df, min_tt_s=None, max_tt_s=None):
    return apply_rules(raw_df, AMBULANCE_UNIT_TYPES, min_tt_s, max_tt_s)
