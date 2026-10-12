"""Stage 2 lookup-table baselines, fitted on train and evaluated on valid (never test).

Run from the project root:  python -m src.baselines
Writes reports/stage2_baselines.txt and echoes it to stdout.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics
from src.clean import load_processed, priority_group
from src.diagnostics import section
from src.splits import assign_time_split, split_counts

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "stage2_baselines.txt"
TARGET = "travel_time_s"
MIN_CELL_ROWS = 30

# Each level is (label, grouping keys), finest first; [] is the global fallback.
L0 = ("B0", [])
L1 = ("B1", ["original_priority_cat"])
L2 = ("B2", ["station_area", "priority_group"])
L3 = ("B3", ["station_area", "priority_group", "hour_band"])
BASELINES = {
    "B0 global median": [L0],
    "B1 priority_cat": [L1, L0],
    "B2 station x prio_grp": [L2, L1, L0],
    "B3 station x prio_grp x hour_band": [L3, L2, L1, L0],
}


def add_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["priority_group"] = priority_group(df["original_priority_cat"])
    # Bands from the request; hour of received_dttm is known before dispatch.
    df["hour_band"] = pd.cut(df["received_dttm"].dt.hour, [-1, 6, 18, 23],
                             labels=["00-06", "07-18", "19-23"]).astype(str)
    return df


class CellQuantile:
    """Quantile of the target per cell, falling back to coarser levels when a cell is thin."""

    def __init__(self, levels, q=0.5, min_n=MIN_CELL_ROWS):
        self.levels, self.q, self.min_n = levels, q, min_n

    def fit(self, df):
        self.tables_ = {}
        for label, keys in self.levels:
            if keys:
                g = df.groupby(keys, observed=True)[TARGET]
                self.tables_[label] = pd.DataFrame({"n": g.size(), "value": g.quantile(self.q)})
            else:
                self.tables_[label] = df[TARGET].quantile(self.q)
        return self

    def predict(self, df):
        """Return (prediction, level label used) per row."""
        pred = pd.Series(np.nan, index=df.index)
        used = pd.Series(pd.NA, index=df.index, dtype="string")
        for label, keys in self.levels:
            todo = pred.isna()
            if not keys:
                # Global value always applies, whatever its n.
                pred[todo], used[todo] = self.tables_[label], label
                break
            # A cell is usable only if it had >= min_n training rows; unseen cells count as 0.
            looked_up = df[keys].merge(self.tables_[label], left_on=keys, right_index=True, how="left")
            ok = todo & (looked_up["n"].fillna(0).to_numpy() >= self.min_n)
            pred[ok], used[ok] = looked_up.loc[ok.to_numpy(), "value"].to_numpy(), label
        return pred, used


def metric_table(y, preds: dict, segment=None) -> pd.DataFrame:
    rows = []
    groups = [("ALL", slice(None))] if segment is None else \
        [(seg, (segment == seg).to_numpy()) for seg in sorted(segment.unique())]
    for seg, mask in groups:
        y_seg = y.to_numpy()[mask]
        for name, p in preds.items():
            rows.append({"segment": seg, "baseline": name, "n": len(y_seg),
                         **metrics.report(y_seg, p.to_numpy()[mask])})
    t = pd.DataFrame(rows).set_index(["segment", "baseline"])
    t[f"within±{metrics.WITHIN_S}s"] = (t[f"within±{metrics.WITHIN_S}s"] * 100).round(1)
    return t.round(1).rename(columns={f"within±{metrics.WITHIN_S}s": f"within±{metrics.WITHIN_S}s_%"})


def run():
    """Fit every baseline on train, predict valid. Returns everything the report/notebook need."""
    df = add_baseline_columns(load_processed())
    df["split"] = assign_time_split(df)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]

    preds, fallbacks = {}, {}
    for name, levels in BASELINES.items():
        preds[name], used = CellQuantile(levels).fit(train).predict(valid)
        fallbacks[name] = used.value_counts()

    lo, _ = CellQuantile(BASELINES["B2 station x prio_grp"], q=0.1).fit(train).predict(valid)
    hi, _ = CellQuantile(BASELINES["B2 station x prio_grp"], q=0.9).fit(train).predict(valid)

    return {
        "df": df, "valid": valid, "preds": preds, "lo": lo, "hi": hi,
        "split_counts": split_counts(df, df["split"]),
        "fallbacks": pd.DataFrame(fallbacks).T.reindex(columns=["B3", "B2", "B1", "B0"]).fillna(0).astype(int),
    }


def interval_table(valid, lo, hi):
    rows = [{"segment": "ALL", **metrics.interval_report(valid[TARGET], lo, hi)}]
    for col in ["unit_type", "priority_group"]:
        for seg, idx in valid.groupby(col).groups.items():
            rows.append({"segment": f"{col}={seg}",
                         **metrics.interval_report(valid.loc[idx, TARGET], lo[idx], hi[idx])})
    t = pd.DataFrame(rows).set_index("segment")
    t["coverage"] = (t["coverage"] * 100).round(1)
    return t.rename(columns={"coverage": "coverage_%"}).round(1)


def report(r):
    valid, y = r["valid"], r["valid"][TARGET]

    section("Data and split (time split on received_dttm, assigned per call_number)")
    print(f"cleaned ambulance rows (R1-R7): {len(r['df']):,}")
    print(r["split_counts"].to_string())
    print("test is held out: not used below")

    section(f"Fallback level used per VALID row (cell needs >= {MIN_CELL_ROWS} train rows)")
    print(r["fallbacks"].to_string())

    section("VALID: overall")
    print(metric_table(y, r["preds"]).to_string())
    for col in ["unit_type", "priority_group"]:
        section(f"VALID: by {col}")
        print(metric_table(y, r["preds"], valid[col]).to_string())

    section("VALID: B2 80% interval (train 10th-90th percentile per B2 cell, same fallback)")
    print(interval_table(valid, r["lo"], r["hi"]).to_string())


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(run())
    REPORT_PATH.write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
