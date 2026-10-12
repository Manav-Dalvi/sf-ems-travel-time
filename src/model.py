"""Stage 3: LightGBM median model, one-time test evaluation, spatial CV.

Run from the project root:  python -m src.model
Writes reports/stage3_results.txt and echoes it to stdout.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance

from src import baselines, metrics
from src.baselines import B2_NAME, BASELINES, CellQuantile, add_baseline_columns, metric_table
from src.clean import build_ambulance_dataset, load_processed
from src.diagnostics import section
from src.features import FEATURES, add_features, check_workload_60m, make_X
from src.load import load_raw
from src.splits import assign_time_split, spatial_block_folds, spatial_folds, split_counts

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "stage3_results.txt"
TARGET = "travel_time_s"
MODEL_NAME = "LightGBM L1"

# L1 loss so the model predicts the conditional median, matching the MAE-optimal baselines.
# MAX_ITER is a cap, not a choice: the iteration count is picked on valid (we check it isn't hit).
MAX_ITER = 3000
BASE_PARAMS = dict(objective="l1", learning_rate=0.05, random_state=0,
                   deterministic=True, force_row_wise=True, verbose=-1)
GRID = [{"num_leaves": nl, "min_child_samples": mcs} for nl in (15, 31, 63) for mcs in (20, 100)]
FEATURES_NO_WORKLOAD = [f for f in FEATURES if f != "workload_60m"]
DIFF_LABELS = ["MAE(B2) - MAE(model)", "RMSE(B2) - RMSE(model)"]


def fit(train, params, n_iter, features=FEATURES, eval_df=None):
    """Fit on `train`; returns (model, categories) so predictions reuse the training levels."""
    X, cats = make_X(train, features)
    model = LGBMRegressor(**BASE_PARAMS, **params, n_estimators=n_iter)
    kw = {}
    if eval_df is not None:
        kw = dict(eval_set=[(make_X(eval_df, features, cats)[0], eval_df[TARGET])], eval_metric="l1")
    model.fit(X, train[TARGET], **kw)
    return model, cats


def predict(model, cats, df, features=FEATURES, n_iter=None):
    return pd.Series(model.predict(make_X(df, features, cats)[0], num_iteration=n_iter), index=df.index)


def select_n_iter(train, valid, params, features=FEATURES):
    """Best iteration by valid MAE over the full curve (equivalent to early stopping without a patience knob)."""
    model, cats = fit(train, params, MAX_ITER, features, eval_df=valid)
    best = int(np.argmin(model.evals_result_["valid_0"]["l1"])) + 1
    mae = metrics.report(valid[TARGET], predict(model, cats, valid, features, best))["MAE"]
    return best, mae


def load_modelling_frame(raw, clean=None):
    clean = load_processed() if clean is None else clean
    df = add_features(add_baseline_columns(clean), raw)
    df["split"] = assign_time_split(df)
    return df


def tune(train, valid):
    rows = []
    for params in GRID:
        best, mae = select_n_iter(train, valid, params)
        rows.append({**params, "best_iter": best, "hit_cap": best == MAX_ITER, "valid_MAE": mae})
    grid = pd.DataFrame(rows)  # pick on unrounded MAE so ties at display precision don't fall to grid order
    chosen = grid.loc[grid["valid_MAE"].idxmin()]
    params = {"num_leaves": int(chosen["num_leaves"]), "min_child_samples": int(chosen["min_child_samples"])}
    return grid, params, int(chosen["best_iter"])


def spatial_cv(trval, params, n_iter, folds, labels):
    """Fit model and B2 on all folds but one, score MAE on the held-out fold.

    `labels` names what each row's held-out group is (spatial block or battalion).
    n_iter is the one chosen on valid, kept fixed. B2_cell_share_% shows how often B2
    still found its station x priority cell in the held-out fold.
    """
    labels = np.asarray(labels)
    rows = []
    for k in np.unique(folds):
        tr, ho = trval[folds != k], trval[folds == k]
        model, cats = fit(tr, params, n_iter)
        b2_pred, b2_used = CellQuantile(BASELINES[B2_NAME]).fit(tr).predict(ho)
        mae_model = metrics.report(ho[TARGET], predict(model, cats, ho))["MAE"]
        mae_b2 = metrics.report(ho[TARGET], b2_pred)["MAE"]
        rows.append({"held_out": ", ".join(map(str, sorted(set(labels[folds == k])))), "rows": len(ho),
                     f"MAE_{MODEL_NAME}": mae_model, "MAE_B2": mae_b2, "B2_minus_model": mae_b2 - mae_model,
                     "B2_cell_share_%": 100 * (b2_used == "B2").mean()})
    table = pd.DataFrame(rows).round(1)
    summary = table[[f"MAE_{MODEL_NAME}", "MAE_B2", "B2_minus_model"]].agg(["mean", "std", "min", "max"]).round(1)
    return table, summary


def battalion_nesting(df):
    """How far station_area is from being nested in battalion (0 / 0% would mean nested)."""
    ct = pd.crosstab(df["station_area"], df["battalion"])
    off_modal = df["battalion"] != df["station_area"].map(ct.idxmax(axis=1))
    return {"station_areas": len(ct), "station_areas_in_>1_battalion": int(((ct > 0).sum(axis=1) > 1).sum()),
            "rows_outside_modal_battalion_%": round(float(100 * off_modal.mean()), 1)}


def run():
    check_workload_60m()
    raw = load_raw()
    df = load_modelling_frame(raw)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    trval = df[df["split"].isin(["train", "valid"])]
    r = {"split_counts": split_counts(df, df["split"]), "valid": valid, "test": test}

    # Part A: revised baselines on valid.
    r["valid_base_preds"], r["valid_base_fallbacks"] = baselines.fit_predict(train, valid)
    r["valid_lo"], r["valid_hi"] = baselines.b2_interval(train, valid)

    # Part C: tuning and ablation, valid only.
    r["grid"], r["params"], r["n_iter"] = tune(train, valid)
    abl_iter, abl_mae = select_n_iter(train, valid, r["params"], FEATURES_NO_WORKLOAD)
    r["ablation"] = pd.DataFrame([
        {"features": "all", "best_iter": r["n_iter"],
         "valid_MAE": r["grid"]["valid_MAE"].min()},
        {"features": "without workload_60m", "best_iter": abl_iter, "valid_MAE": abl_mae},
    ])

    # Part D: the one and only use of test. Everything refit on train+valid.
    model, cats = fit(trval, r["params"], r["n_iter"])
    preds, r["test_fallbacks"] = baselines.fit_predict(trval, test)
    preds[MODEL_NAME] = predict(model, cats, test)
    r["test_preds"] = preds
    # Headline bootstrap resamples calendar days: calls on the same day share weather,
    # traffic and load, so days (not calls) are the more honest independent unit.
    # The call-level version is kept alongside for comparison.
    r["bootstrap_day"], r["bootstrap"] = (
        metrics.paired_cluster_bootstrap(test[TARGET], preds[B2_NAME], preds[MODEL_NAME], groups,
                                         n_boot=1000, seed=0).set_axis(DIFF_LABELS)
        for groups in (test["received_dttm"].dt.date, test["call_number"]))
    r["n_test_days"] = test["received_dttm"].dt.date.nunique()

    # Sensitivity: same fitted models, test rebuilt without R7 (no upper travel-time cap).
    no_r7 = load_modelling_frame(raw, build_ambulance_dataset(raw, max_tt_s=None)[0])
    test_no_r7 = no_r7[no_r7["split"] == "test"]
    preds_no_r7, _ = baselines.fit_predict(trval, test_no_r7)
    preds_no_r7[MODEL_NAME] = predict(model, cats, test_no_r7)
    r["test_no_r7"], r["test_preds_no_r7"] = test_no_r7, preds_no_r7

    X_test = make_X(test, FEATURES, cats)[0]
    imp = permutation_importance(model, X_test, test[TARGET], scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=0)
    # Score is -MAE, so the drop in score is the increase in MAE (seconds).
    r["importance"] = pd.DataFrame({"MAE_increase_s": imp.importances_mean, "std_s": imp.importances_std},
                                   index=FEATURES).sort_values("MAE_increase_s", ascending=False).round(2)

    # Spatial CV on train+valid. Primary: KMeans location blocks, so the held-out
    # fold's area is genuinely absent from training.
    block_folds, r["trval_blocks"], r["block_km"] = spatial_block_folds(trval, n_blocks=10, seed=0, verbose=False)
    centers = r["block_km"].cluster_centers_
    r["block_rows"] = pd.DataFrame({
        "rows": pd.Series(r["trval_blocks"]).value_counts().sort_index(),
        "centroid_lat": centers[:, 1].round(4),
        "centroid_lon": (centers[:, 0] / r["block_km"].lon_scale_).round(4)}).rename_axis("block")
    r["block_table"], r["block_summary"] = spatial_cv(trval, r["params"], r["n_iter"],
                                                      block_folds, r["trval_blocks"])
    # Kept for reference: stability across battalions, not a clean spatial hold-out,
    # because station_area is not nested in battalion (see battalion_nesting).
    r["spatial_table"], r["spatial_summary"] = spatial_cv(trval, r["params"], r["n_iter"],
                                                          spatial_folds(trval, verbose=False), trval["battalion"])
    r["nesting"] = battalion_nesting(trval)
    r["model"], r["cats"], r["trval"] = model, cats, trval
    return r


def r7_comparison(r):
    rows = []
    for name in r["test_preds"]:
        with_r7 = metrics.report(r["test"][TARGET], r["test_preds"][name])
        without = metrics.report(r["test_no_r7"][TARGET], r["test_preds_no_r7"][name])
        rows.append({"model": name, "MAE_R7": with_r7["MAE"], "MAE_noR7": without["MAE"],
                     "RMSE_R7": with_r7["RMSE"], "RMSE_noR7": without["RMSE"],
                     "RMSE_change": without["RMSE"] - with_r7["RMSE"]})
    return pd.DataFrame(rows).set_index("model").round(1)


def report(r):
    section("Features (all known at dispatch) and split")
    print("used:", FEATURES)
    print("workload_60m self-check: passed")
    print(r["split_counts"].to_string())

    section("A. Revised baselines on VALID (B2 = station_area x original_priority_cat)")
    yv = r["valid"][TARGET]
    print("fallback level per valid row:")
    print(r["valid_base_fallbacks"].to_string())
    for seg in [None, "unit_type", "priority_group"]:
        print(f"\n{'overall' if seg is None else 'by ' + seg}:")
        print(metric_table(yv, r["valid_base_preds"], None if seg is None else r["valid"][seg]).to_string())
    print("\nB2 80% interval on valid:")
    print(baselines.interval_table(r["valid"], r["valid_lo"], r["valid_hi"]).to_string())

    section("C. Tuning on VALID (LightGBM, objective=l1, learning_rate=0.05; fit on train)")
    print(r["grid"].round({"valid_MAE": 2}).to_string(index=False))
    print(f"\nchosen: {r['params']}, n_iter={r['n_iter']}")
    print("\nablation (chosen config, iterations re-selected on valid):")
    print(r["ablation"].round({"valid_MAE": 2}).to_string(index=False))

    section(f"D. TEST (fit on train+valid; {len(r['test']):,} rows)")
    yt = r["test"][TARGET]
    print("baseline fallback level per test row:")
    print(r["test_fallbacks"].to_string())
    for seg in [None, "unit_type", "priority_group"]:
        print(f"\n{'overall' if seg is None else 'by ' + seg}:")
        print(metric_table(yt, r["test_preds"], None if seg is None else r["test"][seg]).to_string())

    print_bootstraps(r)

    print(f"\nsensitivity: test without R7 ({len(r['test_no_r7']):,} rows, "
          f"{len(r['test_no_r7']) - len(r['test']):,} more than with R7), overall:")
    print(metric_table(r["test_no_r7"][TARGET], r["test_preds_no_r7"]).to_string())
    print("\nwith vs without R7:")
    print(r7_comparison(r).to_string())

    print("\npermutation importance on test (MAE scoring, 5 repeats, seed 0):")
    print(r["importance"].to_string())

    print_spatial_cv(r)


def print_bootstraps(r):
    print(f"\npaired bootstrap, B2 - model on TEST (1000 resamples, seed 0), positive = model better:")
    print(f"headline: resampling calendar days of received_dttm ({r['n_test_days']} clusters)")
    print(r["bootstrap_day"].round(1).to_string())
    print(f"\nfor comparison: resampling call_number ({r['test']['call_number'].nunique():,} clusters)")
    print(r["bootstrap"].round(1).to_string())


def print_spatial_cv(r, tag="E"):
    section(f"{tag}. Spatial CV on train+valid: leave-one-spatial-block-out (KMeans, 10 blocks, seed 0)")
    print("rows per block and block centroids (KMeans on one point per call, lon scaled by cos(lat)):")
    print(r["block_rows"].to_string())
    print()
    print(r["block_table"].to_string(index=False))
    print("\nacross folds:")
    print(r["block_summary"].to_string())

    section(f"{tag}2. Stability across battalions (not a clean spatial hold-out)")
    print("CAVEAT: station_area is not nested in battalion (train+valid):", r["nesting"])
    print("so held-out battalions share station areas/locations with training.\n")
    print(r["spatial_table"].to_string(index=False))
    print("\nacross folds:")
    print(r["spatial_summary"].to_string())


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
