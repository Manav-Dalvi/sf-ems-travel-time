"""Stage 1 diagnostics: plain-text evidence for choosing cleaning thresholds. Describes only.

Run from the project root:  python -m src.diagnostics
Writes reports/stage1_diagnostics.txt and echoes it to stdout.
"""

import contextlib
import io
from pathlib import Path

import pandas as pd

from src.clean import (AMBULANCE_UNIT_TYPES, DST_WINDOWS_2025, apply_rules,
                       build_ambulance_dataset, overlaps_window, priority_group,
                       travel_time_seconds)
from src.load import DEFAULT_RAW_PATH, load_raw

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "stage1_diagnostics.txt"

SPRING_FORWARD, FALL_BACK = DST_WINDOWS_2025

SHORT_BINS = [0, 10, 20, 30, 60, 90, 120]
SHORT_LABELS = ["1-10", "11-20", "21-30", "31-60", "61-90", "91-120"]
LONG_THRESHOLDS_MIN = [20, 30, 45, 60, 90, 120]

TIME_COLS = ["received_dttm", "response_dttm", "on_scene_dttm"]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def median_p90(df, by):
    g = df.groupby(by)["travel_time_s"]
    return pd.DataFrame({"n": g.size(), "median_s": g.median(), "p90_s": g.quantile(0.9)}).round(1)


def report(raw):
    raw = raw.copy()
    raw["travel_time_s"] = travel_time_seconds(raw)
    raw["overlaps_spring_fwd"] = overlaps_window(raw, *SPRING_FORWARD)
    raw["overlaps_fall_back"] = overlaps_window(raw, *FALL_BACK)

    section("1. RAW rows with travel_time_s < 0 (all unit types)")
    neg = raw[raw["travel_time_s"] < 0].sort_values("received_dttm")
    print(f"count: {len(neg)}")
    print(f"fall-back window used: {FALL_BACK[0]} to {FALL_BACK[1]}")
    print(neg[TIME_COLS + ["unit_type", "travel_time_s", "overlaps_fall_back"]].to_string())

    section("2. RAW rows overlapping the spring-forward window (all unit types)")
    print(f"window: {SPRING_FORWARD[0]} to {SPRING_FORWARD[1]}")
    sf = raw[raw["overlaps_spring_fwd"]].sort_values("received_dttm")
    print(f"count: {len(sf)}")
    print(sf["unit_type"].value_counts().to_string())
    print()
    print(sf[TIME_COLS + ["unit_type", "travel_time_s"]].to_string())
    print("\nfor reference, RAW median travel_time_s by unit_type (all rows):")
    print(raw.groupby("unit_type")["travel_time_s"].median().to_string())

    # Stage 1 evidence was gathered before R6/R7 existed, so keep them off here.
    amb, exclusions = build_ambulance_dataset(raw.drop(columns=["travel_time_s"]),
                                              min_tt_s=None, max_tt_s=None)
    amb["priority_group"] = priority_group(amb["original_priority_cat"])

    section(f"3. AMBULANCES (MEDIC+PRIVATE) after R1-R5: {len(amb):,} rows")

    section("3a. Short travel times: counts per bin (seconds), by unit_type")
    bins = pd.cut(amb["travel_time_s"], SHORT_BINS, labels=SHORT_LABELS)
    print(pd.crosstab(bins, amb["unit_type"], margins=True).to_string())

    section("3b. Long travel times: counts with travel_time_s > threshold")
    above = pd.DataFrame({f"> {m} min": amb["travel_time_s"] > m * 60 for m in LONG_THRESHOLDS_MIN})
    for label, key in [("by unit_type", "unit_type"),
                       ("by original_priority_cat group (3 / 2 / other)", "priority_group")]:
        print(f"\n{label}:")
        table = above.groupby(amb[key]).sum().T
        table["ALL"] = table.sum(axis=1)
        print(table.to_string())
        print("group n:", amb[key].value_counts().to_dict())

    section("3c. Median and p90 travel_time_s")
    hour = amb["received_dttm"].dt.hour.rename("hour_received")
    month = amb["received_dttm"].dt.month.rename("month_received")
    for label, by in [("by original_priority_cat", "original_priority_cat"),
                      ("by hour of received_dttm", hour),
                      ("by month of received_dttm", month),
                      ("by unit_type", "unit_type")]:
        print(f"\n{label}:")
        print(median_p90(amb, by).to_string())

    section("3d. Crosstab unit_type x als_unit")
    print(pd.crosstab(amb["unit_type"], amb["als_unit"], margins=True).to_string())

    section("3e. Rows per call_number (ambulance rows only) and unit_sequence_in_call_dispatch")
    rows_per_call = amb.groupby("call_number").size()
    print(f"unique call_number: {rows_per_call.size:,}")
    print("\nrows per call -> number of calls:")
    print(rows_per_call.value_counts().sort_index().to_string())
    print("\nunit_sequence_in_call_dispatch -> rows:")
    print(amb["unit_sequence_in_call_dispatch"].value_counts().sort_index().to_string())

    section("3f. Exclusion table: build_ambulance_dataset(min_tt_s=None, max_tt_s=None)")
    print(exclusions.to_string(index=False))

    section("4. For comparison only: ENGINE after R2-R5")
    engine, _ = apply_rules(raw.drop(columns=["travel_time_s"]), {"ENGINE"})
    print(median_p90(engine, "unit_type").to_string())

    section("5. Noise floor: share of timestamps with seconds == 0 (R1-R5 rows, all three unit types)")
    print("expected ~1/60 = 1.67% if timestamps are not rounded to the minute")
    share, median_tt = noise_floor(raw.drop(columns=["travel_time_s"]))
    print(share.to_string())
    print("\nmedian travel_time_s by on_scene seconds == 0:")
    print(median_tt.to_string())


def noise_floor(raw):
    """Do timestamps look minute-rounded? With true second resolution, seconds == 0
    should occur ~1/60 = 1.7% of the time. Uses R1-R5 rows of all three unit types
    (no R6/R7, since this check is evidence for them)."""
    df, _ = apply_rules(raw, AMBULANCE_UNIT_TYPES | {"ENGINE"})
    on0 = df["on_scene_dttm"].dt.second.eq(0)
    resp0 = df["response_dttm"].dt.second.eq(0)
    share = pd.DataFrame({
        "n": df.groupby("unit_type").size(),
        "on_scene_sec0_pct": (on0.groupby(df["unit_type"]).mean() * 100).round(2),
        "response_sec0_pct": (resp0.groupby(df["unit_type"]).mean() * 100).round(2),
    })
    median_tt = (df.groupby(["unit_type", on0.rename("on_scene_sec0")])["travel_time_s"]
                 .agg(["size", "median"]).rename(columns={"size": "n", "median": "median_tt_s"}))
    return share, median_tt


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 50)
    raw = load_raw(DEFAULT_RAW_PATH)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(raw)
    REPORT_PATH.write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()


def call_field_consistency(raw):
    """Is battalion an incident field (constant within a call) or a unit field?
    Uses calls with >1 row in the raw file, all unit types."""
    multi = raw[raw.groupby("call_number")["call_number"].transform("size") > 1]
    per_call = multi.groupby("call_number")
    out = {"calls_with_>1_row": per_call.ngroups}
    for col in ["battalion", "station_area"]:
        differs = per_call[col].nunique(dropna=False) > 1
        out[f"{col}_differs_within_call_%"] = round(float(100 * differs.mean()), 2)
    # Context: if battalion were a fixed unit attribute, each unit would have one battalion.
    out["unit_ids_in_>1_battalion_%"] = round(float(100 * (raw.groupby("unit_id")["battalion"].nunique() > 1).mean()), 1)
    return out


def seq_priority_medians(df):
    """Median travel time and n by unit_sequence_in_call_dispatch (1, 2, 3+) x priority_group."""
    seq = df["unit_sequence_in_call_dispatch"].clip(upper=3).map({1: "1", 2: "2", 3: "3+"}).rename("unit_seq")
    g = df.groupby([seq, priority_group(df["original_priority_cat"]).rename("priority_group")])["travel_time_s"]
    return pd.DataFrame({"n": g.size(), "median_s": g.median()}).unstack("priority_group")
