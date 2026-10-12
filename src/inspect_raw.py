"""Stage 0: plain-text inspection report of the raw data. Describes only; filters nothing.

Run from the project root:  python -m src.inspect_raw [path/to/csv]
"""

import sys

import pandas as pd

from src.load import DEFAULT_RAW_PATH, TIMESTAMP_COLS, load_raw

# Rough box around San Francisco's mainland + Treasure Island. Only used to count
# suspicious coordinates here; it is NOT a validated cleaning threshold.
SF_BBOX = {"lon_min": -122.52, "lon_max": -122.35, "lat_min": 37.70, "lat_max": 37.84}

QUANTILES = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]

VALUE_COUNT_COLS = [
    "unit_type",
    "original_priority",
    "priority",
    "final_priority",
    "call_type_group",
    "als_unit",
]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(path):
    df = load_raw(path)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)

    section("Shape")
    print(f"rows: {len(df):,}   columns: {df.shape[1]}")

    section("Columns, dtypes, null counts")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum(),
                        "null_pct": (df.isna().mean() * 100).round(3)}))

    section("Timestamp ranges")
    print(pd.DataFrame({"min": df[TIMESTAMP_COLS].min(), "max": df[TIMESTAMP_COLS].max()}))

    for col in VALUE_COUNT_COLS:
        section(f"Value counts: {col} (NaN included)")
        print(df[col].value_counts(dropna=False).to_string())

    section("Crosstab: original_priority (rows) vs final_priority (cols), NaN included")
    print(pd.crosstab(df["original_priority"].fillna("<NA>"),
                      df["final_priority"].fillna("<NA>"), margins=True))

    section("Calls vs rows")
    rows_per_call = df.groupby("call_number", dropna=False).size()
    print(f"unique call_number: {df['call_number'].nunique(dropna=False):,}")
    print("\nrows per call -> number of calls:")
    print(rows_per_call.value_counts().sort_index().to_string())
    print("\nrows-per-call summary:")
    print(rows_per_call.describe().to_string())

    section("Location")
    missing = df["lon"].isna() | df["lat"].isna()
    outside = ~missing & ~(df["lon"].between(SF_BBOX["lon_min"], SF_BBOX["lon_max"])
                           & df["lat"].between(SF_BBOX["lat_min"], SF_BBOX["lat_max"]))
    print(f"rough SF bbox used: {SF_BBOX}")
    print(f"case_location null:               {df['case_location'].isna().sum():,}")
    print(f"lon/lat missing (null or unparsed): {missing.sum():,}")
    print(f"lon/lat outside rough SF bbox:     {outside.sum():,}")
    if outside.any():
        print("\nsample of out-of-box coordinates:")
        print(df.loc[outside, ["call_number", "unit_id", "case_location"]].head(10).to_string())

    section("Travel time = on_scene_dttm - response_dttm (seconds)")
    travel_s = (df["on_scene_dttm"] - df["response_dttm"]).dt.total_seconds()
    print(f"travel time NaN (a timestamp missing): {travel_s.isna().sum():,}")
    print(f"travel time <= 0:                      {(travel_s <= 0).sum():,}")
    print(f"  of which == 0:                       {(travel_s == 0).sum():,}")
    print(f"  of which <  0:                       {(travel_s < 0).sum():,}")
    print("\n<= 0 counts by unit_type:")
    print((travel_s <= 0).groupby(df["unit_type"], dropna=False).sum().to_string())

    q_all = travel_s.quantile(QUANTILES).rename("ALL")
    q_by_unit = travel_s.groupby(df["unit_type"], dropna=False).quantile(QUANTILES).unstack(0)
    q = pd.concat([q_all, q_by_unit], axis=1)
    q.index = [f"{p * 100:g}%" for p in QUANTILES]
    counts = travel_s.notna().groupby(df["unit_type"], dropna=False).sum()
    print("\nquantiles (non-NaN travel times):")
    print(q.round(1).to_string())
    print("\nnon-NaN n:", {"ALL": int(travel_s.notna().sum()), **counts.astype(int).to_dict()})

    section("Timestamp ordering violations")
    print(f"dispatch_dttm < received_dttm: {(df['dispatch_dttm'] < df['received_dttm']).sum():,}")
    print(f"response_dttm < dispatch_dttm: {(df['response_dttm'] < df['dispatch_dttm']).sum():,}")
    print("(comparisons with a missing timestamp are not counted)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_PATH)
