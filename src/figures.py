"""Stage 6 descriptive figures (no modelling). Each function draws one figure, saves it, and returns it.

Reuses the style and unit colours from src/plots.py so these match the Stage 1-4 figures.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

from src.clean import priority_group
from src.plots import FIG_DIR, NEUTRAL, SEQ_BLUE, UNIT_COLORS

DPI = 200
# Priority groups get their own hues so they are not confused with unit types;
# line styles are a second encoding so identity never relies on colour alone.
PRIORITY_STYLE = {"3": ("#7b3fb8", "-"), "2": ("#c9a227", "--"), "other": ("#8a8984", ":")}
UNIT_STYLE = {"ENGINE": "-", "MEDIC": "--", "PRIVATE": ":"}
PRIORITY_LABEL = {"3": "Code 3", "2": "Code 2", "other": "other"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=DPI)
    return FIG_DIR / name


def _map_aspect(ax, lat):
    # 1 degree of longitude is cos(lat) times shorter than 1 degree of latitude.
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(4))  # full lon labels collide otherwise
    ax.set(xlabel="lon", ylabel="lat")


def _caption(fig, text):
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=9, color=NEUTRAL, wrap=True)


def calls_by_priority(amb):
    """One row per call_number (priority is constant within a call), grouped 3 / 2 / other."""
    calls = amb.sort_values("unit_sequence_in_call_dispatch").drop_duplicates("call_number")
    return calls.assign(priority_group=priority_group(calls["original_priority_cat"]))


def process_timeline():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_axis_off()
    ax.set(xlim=(0, 11), ylim=(0, 3.4))

    stages = [("received", "received_dttm"), ("dispatch", "dispatch_dttm"),
              ("response\n(en route)", "response_dttm"), ("on scene", "on_scene_dttm")]
    xs = [1.2, 4.0, 6.8, 9.6]
    for x, (label, field) in zip(xs, stages):
        ax.add_patch(FancyBboxPatch((x - 0.95, 2.0), 1.9, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="white", edgecolor=NEUTRAL, linewidth=1.2))
        ax.text(x, 2.62, label, ha="center", va="center", fontsize=10.5, color="#0b0b0b")
        ax.text(x, 2.18, field, ha="center", va="center", fontsize=8.5, color=NEUTRAL, family="monospace")

    intervals = [("dispatch delay", False), ("turnout", False), ("travel time  (target)", True)]
    for (x0, x1), (label, is_target) in zip(zip(xs[:-1], xs[1:]), intervals):
        color = UNIT_COLORS["MEDIC"] if is_target else NEUTRAL
        ax.annotate("", xy=(x1, 1.55), xytext=(x0, 1.55),
                    arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 2.4 if is_target else 1.2})
        ax.text((x0 + x1) / 2, 1.2, label, ha="center", va="center", fontsize=10,
                color=color, fontweight="bold" if is_target else "normal")
    ax.text(9.6 - 1.4, 0.82, "travel_time_s = on_scene_dttm - response_dttm", ha="center",
            fontsize=8.5, color=UNIT_COLORS["MEDIC"], family="monospace")

    # Dispatch is the prediction moment: mark it on the timeline itself.
    ax.plot([xs[1], xs[1]], [0.45, 2.0], color="#0b0b0b", linestyle=(0, (2, 2)), linewidth=1)
    ax.text(xs[1], 0.25, "prediction moment = this unit's dispatch", ha="center", fontsize=9.5, color="#0b0b0b")
    ax.set_title("Process timeline for one unit response (schematic, not to scale)", loc="left")
    return _save(fig, "08_process_timeline.png")


def _kde_grid(lon, lat, extent, n=200):
    """Gaussian KDE (Scott's rule) on a regular grid, computed in local km so the kernel is isotropic in distance."""
    lat0 = np.radians(lat.mean())
    to_km = np.array([[111.32 * np.cos(lat0)], [110.57]])
    kde = gaussian_kde(np.vstack([lon, lat]) * to_km, bw_method="scott")
    gx, gy = np.meshgrid(np.linspace(extent[0], extent[1], n), np.linspace(extent[2], extent[3], n))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()]) * to_km).reshape(gx.shape)
    return dens / dens.max()  # scaled to [0, 1] per panel: shows shape, not absolute intensity


def priority_point_kde(calls):
    pad = 0.005
    extent = (calls["lon"].min() - pad, calls["lon"].max() + pad, calls["lat"].min() - pad, calls["lat"].max() + pad)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    counts = {}
    for row, code in enumerate(["3", "2"]):
        sub = calls[calls["original_priority_cat"] == code]
        counts[f"Code {code}"] = len(sub)
        ax_pts, ax_kde = axes[row]
        ax_pts.scatter(sub["lon"], sub["lat"], s=1.5, alpha=0.15, color=PRIORITY_STYLE[code][0], linewidths=0)
        ax_pts.set_title(f"Code {code}: call locations (n = {len(sub):,} calls)", loc="left")
        im = ax_kde.imshow(_kde_grid(sub["lon"].to_numpy(), sub["lat"].to_numpy(), extent),
                           origin="lower", extent=extent, cmap=SEQ_BLUE, vmin=0, vmax=1)
        ax_kde.set_title(f"Code {code}: kernel density (scaled 0-1)", loc="left")
        fig.colorbar(im, ax=ax_kde, shrink=0.8, label="relative density")
        for ax in (ax_pts, ax_kde):
            ax.set(xlim=extent[:2], ylim=extent[2:])
            _map_aspect(ax, calls["lat"])
    _caption(fig, "Ambulance calls by original priority, one point per call_number. Right: kernel density of call "
                  "locations (descriptive smoothing, scaled 0–1 per panel); locations are intersection-level.\n"
                  "Gaussian kernel, bandwidth by Scott's rule (scipy.stats.gaussian_kde), computed in local km.")
    return _save(fig, "09_priority_points_kde.png"), counts


def _ecdf(ax, values_min, color, linestyle, label):
    x = np.sort(values_min)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, linestyle=linestyle,
            linewidth=2, label=label)
    med = np.median(values_min)
    ax.axvline(med, color=color, linestyle=":", linewidth=1.2)
    return med


def travel_time_ecdfs(amb, engine):
    """Returns (path, medians in seconds). Rows are unit responses, as in stage1_diagnostics."""
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    medians = {}
    for unit, df in [("ENGINE", engine), ("MEDIC", amb[amb["unit_type"] == "MEDIC"]),
                     ("PRIVATE", amb[amb["unit_type"] == "PRIVATE"])]:
        med = _ecdf(ax_u, df["travel_time_s"] / 60, UNIT_COLORS[unit], UNIT_STYLE[unit],
                    f"{unit} (n={len(df):,})")
        medians[unit] = med * 60
    groups = priority_group(amb["original_priority_cat"])
    for code, (color, ls) in PRIORITY_STYLE.items():
        df = amb[groups == code]
        med = _ecdf(ax_p, df["travel_time_s"] / 60, color, ls, f"{PRIORITY_LABEL[code]} (n={len(df):,})")
        medians[PRIORITY_LABEL[code]] = med * 60
    ax_u.set(title="By unit type (ENGINE after R2-R5, for comparison)", ylabel="share of responses ≤ x")
    ax_p.set(title="Ambulances by original priority")
    for ax in (ax_u, ax_p):
        ax.set(xlim=(0, 30), ylim=(0, 1.01), xlabel="travel time (minutes)")
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _caption(fig, "Empirical CDFs of travel time; dotted vertical lines mark each group's median. "
                  "x-axis truncated at 30 minutes.")
    return _save(fig, "10_travel_time_ecdfs.png"), medians


def choose_hex_gridsize(amb, min_count=30, candidates=range(10, 81, 2)):
    """Largest gridsize whose median non-empty hex still holds >= min_count responses."""
    extent = (amb["lon"].min(), amb["lon"].max(), amb["lat"].min(), amb["lat"].max())
    best = None
    for g in candidates:
        fig, ax = plt.subplots()
        # compressed(): drop masked (empty) hexes; np.median would ignore the mask.
        counts = ax.hexbin(amb["lon"], amb["lat"], gridsize=g, extent=extent, mincnt=1).get_array().compressed()
        plt.close(fig)
        if np.median(counts) >= min_count:
            best = (g, float(np.median(counts)))
    return best


def median_travel_hexbin(amb, gridsize, min_count=30):
    fig, ax = plt.subplots(figsize=(8, 6.8))
    hb = ax.hexbin(amb["lon"], amb["lat"], C=amb["travel_time_s"] / 60, reduce_C_function=np.median,
                   gridsize=gridsize, mincnt=min_count, cmap=SEQ_BLUE, linewidths=0.2, edgecolors="white")
    fig.colorbar(hb, ax=ax, shrink=0.7, label="median travel time (minutes)")
    _map_aspect(ax, amb["lat"])
    ax.set_title(f"Median ambulance travel time by hexagon (gridsize = {gridsize})", loc="left")
    _caption(fig, f"Only hexagons with ≥ {min_count} ambulance responses are shown; "
                  f"hexagons with fewer responses are hidden ({len(hb.get_array()):,} hexagons shown).")
    return _save(fig, "11_median_travel_hexbin.png")
