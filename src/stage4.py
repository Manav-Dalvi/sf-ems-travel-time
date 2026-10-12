"""Stage 4: battalion diagnostics, spatial block CV, day bootstrap, CQR intervals, error analysis.

Run from the project root:  python -m src.stage4
Writes reports/stage4_results.txt and echoes it to stdout. Reuses the Stage 3 point
model and baselines unchanged (via model.run()).
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines, intervals, metrics, model
from src.baselines import B2_NAME
from src.diagnostics import call_field_consistency, section, seq_priority_medians
from src.load import load_raw
from src.model import MODEL_NAME, TARGET
from src.splits import assign_spatial_blocks

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "stage4_results.txt"
MIN_STATION_ROWS = 200


def hour_band(received: pd.Series) -> pd.Series:
    return pd.cut(received.dt.hour, [-1, 6, 18, 23], labels=["00-06", "07-18", "19-23"]).astype(str)


def run():
    r = model.run()
    raw = load_raw()
    trval, valid, test = r["trval"], r["valid"], r["test"]
    train = trval[trval["split"] == "train"]

    # Part A diagnostics.
    r["call_fields"] = call_field_consistency(raw)
    r["seq_prio"] = seq_priority_medians(pd.concat([trval, test]))

    # Part D: quantile models on TRAIN only; valid picks iterations and then calibrates.
    qmodels = [intervals.fit_quantile(train, valid, r["params"], a) for a in intervals.ALPHAS]
    r["q_iters"] = {a: m[2] for a, m in zip(intervals.ALPHAS, qmodels)}
    lo_v, hi_v, r["crossed_valid"] = intervals.predict_interval(qmodels, valid)
    lo_t, hi_t, r["crossed_test"] = intervals.predict_interval(qmodels, test)
    r["raw_interval"] = pd.DataFrame(
        [{"split": name, "n": len(d), **metrics.interval_report(d[TARGET], lo, hi)}
         for name, d, lo, hi in [("valid", valid, lo_v, hi_v), ("test", test, lo_t, hi_t)]]).set_index("split")

    r["cqr_q"], r["cqr_level"], r["cqr_n"] = intervals.cqr_margin(valid[TARGET], lo_v, hi_v)
    cqr_lo, cqr_hi = lo_t - r["cqr_q"], hi_t + r["cqr_q"]
    b2_lo, b2_hi = baselines.b2_interval(trval, test)  # refit on train+valid, as in Stage 3
    r["interval_test"] = intervals.interval_compare(
        test, {"raw_QR": (lo_t, hi_t), "CQR": (cqr_lo, cqr_hi), "B2": (b2_lo, b2_hi)})

    # Part E: error analysis on test.
    t = test.copy()
    t["pred"] = r["test_preds"][MODEL_NAME]
    t["abs_err"] = (t[TARGET] - t["pred"]).abs()
    t["covered"] = (t[TARGET] >= cqr_lo) & (t[TARGET] <= cqr_hi)
    t["width"] = cqr_hi - cqr_lo
    t["hour_band"] = hour_band(t["received_dttm"])
    # Blocks come from the KMeans fitted on train+valid locations in model.run().
    t["spatial_block"] = assign_spatial_blocks(r["block_km"], t)
    r["errors_by"] = {seg: error_table(t, seg)
                      for seg in ["priority_group", "unit_type", "hour_band", "spatial_block"]}

    by_station = error_table(t, "station_area")
    eligible = by_station[by_station["n"] >= MIN_STATION_ROWS].sort_values("MAE")
    r["station_eligible"] = (len(eligible), len(by_station))
    r["station_best"], r["station_worst"] = eligible.head(5), eligible.tail(5).iloc[::-1]

    r["tail_threshold"] = float(t[TARGET].quantile(0.9))
    tail = t[t[TARGET] > r["tail_threshold"]]
    rows = []
    for name, pred in [(MODEL_NAME, tail["pred"]), (B2_NAME, r["test_preds"][B2_NAME][tail.index])]:
        err = tail[TARGET] - pred  # positive = under-prediction
        rows.append({"model": name, "n": len(tail), "MAE": err.abs().mean(), "mean_signed_error": err.mean(),
                     "under_predicted_%": 100 * (err > 0).mean()})
    r["tail"] = pd.DataFrame(rows).set_index("model").round(1)
    r["test_frame"] = t
    return r


def error_table(t, seg):
    g = t.groupby(seg)
    return pd.DataFrame({
        "n": g.size(),
        "MAE": g["abs_err"].mean(),
        "P90AE": g["abs_err"].quantile(0.9),
        "CQR_cov_%": 100 * g["covered"].mean(),
        "CQR_width_s": g["width"].mean(),
    }).round(1)


def report(r):
    section("A1. Is battalion an incident or a unit field? (RAW, calls with >1 row)")
    for k, v in r["call_fields"].items():
        print(f"{k}: {v}")

    section("A2. Cleaned ambulance data (all splits): travel_time_s by unit_seq x priority_group")
    print(r["seq_prio"].to_string())

    model.print_spatial_cv(r, tag="B")

    section("C. Paired bootstrap on TEST")
    model.print_bootstraps(r)

    section("D. 80% prediction intervals")
    print(f"quantile models: LightGBM objective=quantile, {r['params']}, learning_rate=0.05, fit on train")
    print(f"iterations chosen on valid pinball loss: {r['q_iters']}")
    print(f"crossed lo > hi (reordered): valid {r['crossed_valid']}, test {r['crossed_test']}")
    print("\n1. raw quantile intervals:")
    print(r["raw_interval"].round({"coverage": 3, "mean_width_s": 1}).to_string())
    print(f"\n2. CQR on valid: n={r['cqr_n']:,}, quantile level={r['cqr_level']:.5f}, q={r['cqr_q']:.1f} s")
    print("   final interval = [lo - q, hi + q]")
    print("\n3. TEST comparison (B2 = train+valid 10th-90th pct per B2 cell):")
    print(r["interval_test"].to_string())

    section("E. Error analysis on TEST (LightGBM L1 point model, CQR intervals)")
    for seg, table in r["errors_by"].items():
        print(f"\nby {seg}:")
        print(table.to_string())
    n_ok, n_all = r["station_eligible"]
    print(f"\nstation_area with >= {MIN_STATION_ROWS} test rows: {n_ok} of {n_all}")
    print("lowest MAE (best 5):")
    print(r["station_best"].to_string())
    print("highest MAE (worst 5):")
    print(r["station_worst"].to_string())
    print(f"\ntail: rows with actual > test 90th percentile ({r['tail_threshold']:.0f} s); "
          "signed error = actual - predicted (positive = under-predicts)")
    print(r["tail"].to_string())


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(run())
    REPORT_PATH.write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
