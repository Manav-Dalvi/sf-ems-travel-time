"""Stage 7 (supplementary): is travel time conditionally independent of turnout given Z?

Run from the project root:  python -m src.turnout_ci
Writes reports/stage7_turnout_ci.txt and a figure, and echoes the report to stdout.

Uses train + valid (received Jan-Sep 2025) only; the test period is never loaded into
any fit or statistic. Turnout (response_dttm - dispatch_dttm) is NOT a dispatch-time
feature: it is used here only for this check, never in the main model.
"""

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm, rankdata
from sklearn.model_selection import GroupKFold

from src import metrics
from src.clean import load_processed
from src.diagnostics import section
from src.features import FEATURES, check_workload_60m, make_X
from src.load import load_raw
from src.model import BASE_PARAMS, TARGET, load_modelling_frame
from src.plots import FIG_DIR, NEUTRAL, UNIT_COLORS

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "stage7_turnout_ci.txt"
FIG_NAME = "12_turnout_residual_deciles.png"

TEST_START = pd.Timestamp("2025-10-01")  # first day of the test period (src.splits.SPLIT_RANGES)
N_FOLDS, N_BOOT, SEED = 5, 1000, 0
CUTOFF_QUANTILES = [0.99, 0.999]  # upper turnout cutoffs, run as a sensitivity
TURNOUT_QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]

# Pre-specified (before results): large n makes tiny effects significant, so these
# thresholds, not p-values, decide "practically negligible".
MAX_ABS_PARTIAL_CORR = 0.05
MAX_MAE_GAIN_S = 2.0

# Residual models for the double-residual check: fixed, no tuning.
RESID_PARAMS = dict(BASE_PARAMS, objective="l2", num_leaves=15, min_child_samples=20, n_estimators=1000)
# Stage 3 point model: chosen config and n_iter from reports/stage3_results.txt (section C).
STAGE3_N_ITER = 1519
POINT_PARAMS = dict(BASE_PARAMS, num_leaves=15, min_child_samples=20, n_estimators=STAGE3_N_ITER)

SEGMENTS = [("overall", None, None)] + \
           [(f"priority {g}", "priority_group", g) for g in ("3", "2", "other")] + \
           [(u, "unit_type", u) for u in ("MEDIC", "PRIVATE")]


# ---------------------------------------------------------------- data

def load_trval():
    """Cleaned ambulance rows (R1-R7) received Jan-Sep 2025, with Stage 3 features and turnout_s."""
    check_workload_60m()
    clean = load_processed()
    clean = clean[clean["received_dttm"] < TEST_START]
    df = load_modelling_frame(load_raw(), clean)
    assert df["split"].isin(["train", "valid"]).all(), "test rows present"
    df["turnout_s"] = (df["response_dttm"] - df["dispatch_dttm"]).dt.total_seconds()
    df["day"] = df["received_dttm"].dt.date
    iso = df["received_dttm"].dt.isocalendar()
    df["iso_week"] = (iso["year"] * 100 + iso["week"]).astype(int)
    return df.reset_index(drop=True)


def week_folds(df):
    """5-fold GroupKFold on ISO week of received_dttm; a call has one received time, so it stays in one fold."""
    folds = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["iso_week"])):
        folds[idx] = k
    assert (pd.Series(folds).groupby(df["call_number"].to_numpy()).nunique() == 1).all(), \
        "a call_number sits in more than one fold"
    return folds


# ---------------------------------------------------------------- fitting

def oof_predict(df, y, features, params, folds):
    pred = np.empty(len(df))
    for k in np.unique(folds):
        tr, ho = folds != k, folds == k
        X_tr, cats = make_X(df[tr], features)
        model = LGBMRegressor(**params).fit(X_tr, y[tr])
        pred[ho] = model.predict(make_X(df[ho], features, cats)[0])
    return pred


def double_residuals(df):
    """OOF residuals of log1p(travel) ~ Z and log1p(turnout) ~ Z."""
    folds = week_folds(df)
    out = {}
    for name, col in [("r_Y", TARGET), ("r_X", "turnout_s")]:
        y = np.log1p(df[col].to_numpy())
        out[name] = y - oof_predict(df, y, FEATURES, RESID_PARAMS, folds)
    return out["r_Y"], out["r_X"]


# ---------------------------------------------------------------- statistics

def _corr(a, b):
    # Elementwise sums rather than np.corrcoef: macOS Accelerate BLAS warns spuriously on matmul.
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())


def dependence_stats(r_y, r_x):
    prod = r_y * r_x
    return np.array([_corr(r_y, r_x), _corr(rankdata(r_y), rankdata(r_x)), prod.mean()])


def day_bootstrap(stat_fn, arrays, days, n_boot=N_BOOT, seed=SEED):
    """Point estimate and 95% percentile CI, resampling whole calendar days with replacement.

    Calls on one day share weather, traffic and load, so days are the independent unit.
    """
    codes, uniques = pd.factorize(np.asarray(days))
    order = np.argsort(codes, kind="stable")
    by_day = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([by_day[g] for g in rng.integers(0, len(uniques), len(uniques))])
        boot.append(stat_fn(*(a[idx] for a in arrays)))
    boot = np.asarray(boot)
    return stat_fn(*arrays), np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def gcm_T(r_y, r_x):
    # Asymptotic N(0,1) null assumes independent rows; rows share days (and calls), so
    # the day-bootstrap CI for mean(r_Y * r_X) is the headline, not this p-value.
    prod = r_y * r_x
    t = np.sqrt(len(prod)) * prod.mean() / prod.std(ddof=1)
    return t, 2 * norm.sf(abs(t))


def dependence_row(case, r_y, r_x, days):
    point, lo, hi = day_bootstrap(dependence_stats, (r_y, r_x), days)
    t, p = gcm_T(r_y, r_x)
    return {"case": case, "n": len(r_y), "days": pd.Series(days).nunique(),
            "pearson": point[0], "pearson_lo": lo[0], "pearson_hi": hi[0],
            "spearman": point[1], "spearman_lo": lo[1], "spearman_hi": hi[1],
            "mean_rYrX": point[2], "mean_lo": lo[2], "mean_hi": hi[2],
            "GCM_T": t, "GCM_p_iid": p}


def segment_masks(df):
    for label, col, val in SEGMENTS:
        yield label, np.ones(len(df), bool) if col is None else (df[col] == val).to_numpy()


# ---------------------------------------------------------------- parts

def part_b(df, r_y, r_x):
    days = df["day"].to_numpy()
    return pd.DataFrame([dependence_row(label, r_y[m], r_x[m], days[m]) for label, m in segment_masks(df)])


def part_c(df):
    folds = week_folds(df)
    y = df[TARGET].to_numpy()
    without = oof_predict(df, y, FEATURES, POINT_PARAMS, folds)
    with_t = oof_predict(df, y, FEATURES + ["turnout_s"], POINT_PARAMS, folds)
    days = df["day"].to_numpy()
    gain_rows, spear_rows = [], []
    for label, m in segment_masks(df):
        boot = metrics.paired_cluster_bootstrap(y[m], without[m], with_t[m], days[m], n_boot=N_BOOT, seed=SEED)
        mae = boot.loc["MAE(a) - MAE(b)"]
        gain_rows.append({"case": label, "n": int(m.sum()),
                          "MAE_without": np.abs(without[m] - y[m]).mean(),
                          "MAE_with": np.abs(with_t[m] - y[m]).mean(),
                          "MAE_gain": mae["estimate"], "gain_lo": mae["ci_2.5%"], "gain_hi": mae["ci_97.5%"]})
        if label == "overall" or label.startswith("priority"):
            resid = y[m] - without[m]  # actual - predicted
            point, lo, hi = day_bootstrap(lambda a, b: _corr(rankdata(a), rankdata(b)),
                                          (resid, df["turnout_s"].to_numpy()[m]), days[m])
            spear_rows.append({"case": label, "n": int(m.sum()), "spearman": point,
                               "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(gain_rows), pd.DataFrame(spear_rows)


def part_d(df):
    """Recording-artefact checks: refit both residual models within each subset, overall only."""
    cases = [("(i) travel_time_s >= 60", df[TARGET] >= 60),
             ("(ii) turnout_s >= 15", df["turnout_s"] >= 15),
             ("(iii) both", (df[TARGET] >= 60) & (df["turnout_s"] >= 15)),
             ("(iv) MEDIC only", df["unit_type"] == "MEDIC"),
             ("(iv) PRIVATE only", df["unit_type"] == "PRIVATE")]
    rows = []
    for label, keep in cases:
        sub = df[keep].reset_index(drop=True)
        r_y, r_x = double_residuals(sub)
        rows.append(dependence_row(label, r_y, r_x, sub["day"].to_numpy()))
    return pd.DataFrame(rows)


def decile_curve(r_y, r_x, days):
    """Median r_Y per decile of r_X (deciles fixed on the full segment), with day-bootstrap 95% bands."""
    bins = pd.qcut(r_x, 10, labels=False)
    x_mid = pd.Series(r_x).groupby(bins).median().to_numpy()

    def medians(ry, b):
        return pd.Series(ry).groupby(b).median().reindex(range(10)).to_numpy()

    point, lo, hi = day_bootstrap(medians, (r_y, bins), days)
    return x_mid, point, lo, hi


def part_e(df, r_y, r_x, cutoff_label):
    days = df["day"].to_numpy()
    panels = [m for m in segment_masks(df)][:4]  # overall + 3 priority groups
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, (label, m) in zip(axes, panels):
        x, med, lo, hi = decile_curve(r_y[m], r_x[m], days[m])
        ax.fill_between(x, lo, hi, color=UNIT_COLORS["MEDIC"], alpha=0.2, linewidth=0, label="95% band (day bootstrap)")
        ax.plot(x, med, color=UNIT_COLORS["MEDIC"], marker="o", markersize=4, linewidth=2, label="median r_Y")
        ax.axhline(0, color=NEUTRAL, linewidth=0.8)
        ax.set(title=f"{label} (n={int(m.sum()):,})", xlabel="r_X (decile median)")
    axes[0].set_ylabel("median r_Y")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle(f"OOF residuals: log1p(travel) vs log1p(turnout), each given Z; turnout cutoff {cutoff_label}",
                 x=0.01, ha="left")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / FIG_NAME, dpi=200)
    return fig


# ---------------------------------------------------------------- run / report

def run():
    df = load_trval()
    r = {"n_trval": len(df), "days": df["day"].nunique(), "weeks": df["iso_week"].nunique()}

    nonpos = df["turnout_s"] <= 0
    r["nonpos"] = pd.DataFrame({"turnout_s <= 0": df[nonpos].groupby("unit_type").size(),
                                "turnout_s == 0": df[df["turnout_s"] == 0].groupby("unit_type").size(),
                                "turnout_s < 0": df[df["turnout_s"] < 0].groupby("unit_type").size()}
                               ).reindex(["MEDIC", "PRIVATE"]).fillna(0).astype(int)
    base = df[~nonpos].reset_index(drop=True)
    r["n_base"] = len(base)
    r["quantiles"] = base.groupby("unit_type")["turnout_s"].quantile(TURNOUT_QUANTILES).unstack()
    r["quantiles"].loc["ALL"] = base["turnout_s"].quantile(TURNOUT_QUANTILES)

    r["cutoffs"] = {}
    for q in CUTOFF_QUANTILES:
        label = f"p{100 * q:g}"
        cut = base["turnout_s"].quantile(q)
        d = base[base["turnout_s"] <= cut].reset_index(drop=True)
        r_y, r_x = double_residuals(d)
        gains, spear = part_c(d)
        r["cutoffs"][label] = {
            "value": cut, "removed": int((base["turnout_s"] > cut).sum()),
            "removed_by_unit": base[base["turnout_s"] > cut].groupby("unit_type").size().to_dict(),
            "n": len(d), "folds": pd.Series(week_folds(d)).value_counts().sort_index(),
            "B": part_b(d, r_y, r_x), "C_gain": gains, "C_spearman": spear,
        }
        if q == 0.999:
            r["D"] = part_d(d)
            part_e(d, r_y, r_x, f"{label} = {cut:.0f} s")
    return r


def _fmt_dependence(t):
    out = pd.DataFrame({
        "case": t["case"], "n": t["n"].map("{:,}".format), "days": t["days"],
        "pearson [95% CI]": [f"{p:+.4f} [{a:+.4f}, {b:+.4f}]" for p, a, b in zip(t["pearson"], t["pearson_lo"], t["pearson_hi"])],
        "spearman [95% CI]": [f"{p:+.4f} [{a:+.4f}, {b:+.4f}]" for p, a, b in zip(t["spearman"], t["spearman_lo"], t["spearman_hi"])],
        "mean(r_Y*r_X) [95% CI]": [f"{p:+.5f} [{a:+.5f}, {b:+.5f}]" for p, a, b in zip(t["mean_rYrX"], t["mean_lo"], t["mean_hi"])],
        "GCM T": t["GCM_T"].round(2), "p (iid N(0,1))": t["GCM_p_iid"].map("{:.2g}".format),
    })
    return out.to_string(index=False)


def _sign(lo, hi):
    return "positive" if lo > 0 else "negative" if hi < 0 else "CI includes 0"


def _verdict(corr, gain=None):
    corr_ok = abs(corr) < MAX_ABS_PARTIAL_CORR
    if gain is None:
        return (f"|partial corr| = {abs(corr):.4f} ({'<' if corr_ok else '>='} {MAX_ABS_PARTIAL_CORR}); "
                f"MAE criterion not evaluated -> {'negligible' if corr_ok else 'not negligible'} (correlation criterion only)")
    gain_ok = gain < MAX_MAE_GAIN_S
    ok = corr_ok and gain_ok
    return (f"|partial corr| = {abs(corr):.4f} ({'<' if corr_ok else '>='} {MAX_ABS_PARTIAL_CORR}); "
            f"MAE gain = {gain:+.2f} s ({'<' if gain_ok else '>='} {MAX_MAE_GAIN_S:g}) -> "
            f"{'negligible' if ok else 'not negligible'}")


def thresholds_text():
    return (f'"Practically negligible" = |partial correlation| < {MAX_ABS_PARTIAL_CORR} AND OOF MAE gain from '
            f"adding turnout < {MAX_MAE_GAIN_S:g} s.\n"
            "partial correlation = point estimate of the Pearson correlation of the OOF residuals r_Y, r_X "
            "(log1p scale); MAE gain = MAE(without turnout) - MAE(with turnout), OOF, point estimate.\n"
            "Large n makes tiny effects statistically significant; these thresholds, not p-values, decide.")


def report(r):
    section("Stage 7 (supplementary): conditional independence of travel time and turnout given Z")
    print("data: cleaned ambulance rows (R1-R7), received Jan-Sep 2025 (train + valid); test period not used")
    print(f"rows: {r['n_trval']:,}   calendar days: {r['days']}   ISO weeks: {r['weeks']}")
    print("Z (Stage 3 features):", FEATURES)
    print("turnout_s = response_dttm - dispatch_dttm (not a dispatch-time feature; used only for this check)")
    print(f"folds: {N_FOLDS}-fold GroupKFold, groups = ISO week of received_dttm; all predictions OOF")
    print(f"bootstrap: {N_BOOT} resamples of calendar days, seed {SEED}")
    print("\npre-specified thresholds:\n" + thresholds_text())

    section("A. Turnout: non-positive values, quantiles, upper cutoffs")
    print("dropped turnout_s <= 0, by unit_type:")
    print(r["nonpos"].to_string())
    print(f"total dropped: {r['nonpos']['turnout_s <= 0'].sum():,}   rows kept: {r['n_base']:,}")
    print("\nturnout_s quantiles (s), after dropping <= 0:")
    print(r["quantiles"].rename(columns=lambda q: f"{100 * q:g}%").round(1).to_string())
    print("\nupper cutoffs (sensitivity, both run in full):")
    for label, c in r["cutoffs"].items():
        print(f"  {label}: turnout_s <= {c['value']:.1f} s; removed {c['removed']:,} rows "
              f"{c['removed_by_unit']}; rows used: {c['n']:,}")
    for label, c in r["cutoffs"].items():
        print(f"  rows per fold at {label}: {c['folds'].to_dict()}")

    for label, c in r["cutoffs"].items():
        section(f"B. Dependence of OOF residuals (double residual / GCM-style), cutoff {label} = {c['value']:.1f} s")
        print("r_Y = OOF residual of log1p(travel_time_s) ~ Z; r_X = OOF residual of log1p(turnout_s) ~ Z")
        print(f"residual models: LightGBM {dict((k, RESID_PARAMS[k]) for k in ('objective', 'num_leaves', 'min_child_samples', 'learning_rate', 'n_estimators'))}")
        print("CIs: day bootstrap. GCM T = sqrt(n) mean(r_Y r_X) / sd(r_Y r_X); its N(0,1) p-value assumes "
              "independent rows, so the day-bootstrap CI for mean(r_Y r_X) is the headline.")
        print(_fmt_dependence(c["B"]))

        section(f"C. Practical relevance, cutoff {label}: Stage 3 point model with vs without turnout_s")
        print(f"LightGBM L1, num_leaves 15, min_child_samples 20, lr 0.05, n_estimators {STAGE3_N_ITER} (Stage 3); OOF")
        g = c["C_gain"]
        print(pd.DataFrame({
            "case": g["case"], "n": g["n"].map("{:,}".format),
            "MAE without": g["MAE_without"].round(2), "MAE with": g["MAE_with"].round(2),
            "gain (without - with) [95% CI]": [f"{p:+.2f} [{a:+.2f}, {b:+.2f}]"
                                               for p, a, b in zip(g["MAE_gain"], g["gain_lo"], g["gain_hi"])],
        }).to_string(index=False))
        print("\nSpearman(OOF residual of WITHOUT model, actual - predicted; turnout_s):")
        s = c["C_spearman"]
        print(pd.DataFrame({"case": s["case"], "n": s["n"].map("{:,}".format),
                            "spearman [95% CI]": [f"{p:+.4f} [{a:+.4f}, {b:+.4f}]"
                                                  for p, a, b in zip(s["spearman"], s["ci_lo"], s["ci_hi"])]
                            }).to_string(index=False))

    p999 = r["cutoffs"]["p99.9"]
    section(f"D. Recording-artefact checks (Part B overall, residual models refit per subset; cutoff p99.9 = {p999['value']:.1f} s)")
    print("expected signs: late 'en route' press -> longer turnout, shorter travel (negative);")
    print("unit away from base -> both longer (positive)")
    print(_fmt_dependence(r["D"]))
    print("\nsign of mean(r_Y*r_X) by its day-bootstrap 95% CI / sign of Pearson point estimate:")
    for _, row in r["D"].iterrows():
        print(f"  {row['case']}: {_sign(row['mean_lo'], row['mean_hi'])}; Pearson {row['pearson']:+.4f}")

    section("E. Figure")
    print(f"reports/figures/{FIG_NAME}: median r_Y by decile of r_X, overall and by priority_group, "
          f"cutoff p99.9; 95% bands from the day bootstrap")

    section("Pre-specified thresholds and verdicts")
    print(thresholds_text())
    for label, c in r["cutoffs"].items():
        print(f"\ncutoff {label}:")
        gains = c["C_gain"].set_index("case")["MAE_gain"]
        for _, row in c["B"].iterrows():
            print(f"  {row['case']}: {_verdict(row['pearson'], gains[row['case']])}")
    print("\nPart D (cutoff p99.9):")
    for _, row in r["D"].iterrows():
        print(f"  {row['case']}: {_verdict(row['pearson'])}")


def main():
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(run())
    REPORT_PATH.write_text(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
