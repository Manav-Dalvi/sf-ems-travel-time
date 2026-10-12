"""Stage 1 EDA figures. Each function draws one figure, saves it to reports/figures/, and returns it."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Fixed colour per unit type so the same entity looks the same in every figure.
UNIT_COLORS = {"MEDIC": "#2a78d6", "PRIVATE": "#eb6834", "ENGINE": "#1baf7a"}
NEUTRAL = "#52514e"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b0afaa", "axes.labelcolor": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "xtick.color": NEUTRAL, "ytick.color": NEUTRAL, "font.size": 10,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    return fig


def travel_time_hist(amb):
    tt = amb["travel_time_s"]
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4))

    ax_lin.hist(tt, bins=np.arange(0, 3600 + 30, 30), range=(0, 3600), color=NEUTRAL)
    ax_lin.set(title=f"Ambulance travel time, linear (0-3600 s; {(tt > 3600).sum()} rows > 3600 s not shown)",
               xlabel="travel_time_s", ylabel="rows")

    # Log-spaced bins so each bin is equally wide on the log axis.
    ax_log.hist(tt, bins=np.logspace(0, np.log10(tt.max()), 80), color=NEUTRAL)
    ax_log.set_xscale("log")
    ax_log.set(title="Ambulance travel time, log x-axis (all rows)", xlabel="travel_time_s (log)", ylabel="rows")
    return _save(fig, "01_travel_time_hist.png")


def travel_time_density_by_unit(amb, engine):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(0, 3600 + 30, 30)
    for unit, df in [("MEDIC", amb[amb["unit_type"] == "MEDIC"]),
                     ("PRIVATE", amb[amb["unit_type"] == "PRIVATE"]),
                     ("ENGINE", engine)]:
        ax.hist(df["travel_time_s"], bins=bins, density=True, histtype="step", linewidth=2,
                color=UNIT_COLORS[unit], label=f"{unit} (n={len(df):,})")
    ax.set(title="Travel time density by unit type (0-3600 s; ENGINE after R2-R5, for comparison)",
           xlabel="travel_time_s", ylabel="density")
    ax.legend(frameon=False)
    return _save(fig, "02_travel_time_density_by_unit.png")


def median_by_hour(amb):
    med = amb.groupby([amb["received_dttm"].dt.hour, "unit_type"])["travel_time_s"].median().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    for unit in med.columns:
        ax.plot(med.index, med[unit], color=UNIT_COLORS[unit], linewidth=2, marker="o", markersize=4, label=unit)
        ax.annotate(unit, (med.index[-1], med[unit].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=NEUTRAL)
    ax.set(title="Median ambulance travel time by hour of received_dttm",
           xlabel="hour of day (local)", ylabel="median travel_time_s", xticks=range(0, 24))
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "03_median_by_hour.png")


def boxplot_by_priority(amb):
    cats = sorted(amb["original_priority_cat"].unique(), key=lambda c: (c == "missing", c))
    data = [amb.loc[amb["original_priority_cat"] == c, "travel_time_s"] for c in cats]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # Whiskers at 5th/95th percentiles; outliers hidden because the long tail would flatten the boxes.
    ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.6,
               medianprops={"color": "#0b0b0b"}, boxprops={"color": NEUTRAL},
               whiskerprops={"color": NEUTRAL}, capprops={"color": NEUTRAL})
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels([f"{c}\nn={len(d):,}" for c, d in zip(cats, data)])
    ax.set(title="Ambulance travel time by original_priority_cat (box = IQR, whiskers = 5th-95th pct)",
           xlabel="original_priority_cat", ylabel="travel_time_s")
    return _save(fig, "04_boxplot_by_priority.png")


def box_median_map(amb):
    per_box = amb.groupby("box").agg(lon=("lon", "median"), lat=("lat", "median"),
                                     median_tt=("travel_time_s", "median"), n=("travel_time_s", "size"))
    # Clip the colour scale at the 5th/95th pct of box medians so a few extreme
    # (often low-n) boxes don't wash out the rest.
    vmin, vmax = per_box["median_tt"].quantile([0.05, 0.95])
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(per_box["lon"], per_box["lat"], c=per_box["median_tt"], cmap=SEQ_BLUE,
                    vmin=vmin, vmax=vmax, s=14, linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="median travel_time_s (clipped at 5th-95th pct of boxes)")
    # Equal-distance aspect at SF's latitude so the city isn't stretched.
    ax.set_aspect(1 / np.cos(np.radians(per_box["lat"].median())))
    ax.grid(False)
    ax.set(title=f"Median ambulance travel time per box ({len(per_box):,} boxes; point = median lon/lat)",
           xlabel="lon", ylabel="lat")
    return _save(fig, "05_box_median_map.png")
