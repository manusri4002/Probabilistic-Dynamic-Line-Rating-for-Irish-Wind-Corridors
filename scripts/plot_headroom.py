"""
Rating duration curve and monthly headroom distributions for the north Dublin corridor.

Everything plotted depends on the assumed line bearing and sheltering factor, so the duration
curve shows the central case together with the range over the plausible sheltering factors and
the worst bearing of the sweep (dlr/bands.py). The period in the titles comes from the data.
"""

import calendar
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from dlr import config
from dlr.bands import scenario_table, worst_span
from dlr.pipeline import ROOT_DIR, add_dlr_columns, load_weather

OUTPUT_NAME = "dlr_headroom_duration_and_monthly.png"


def duration_curve(ratio):
    """x: percentage of time the ratio is exceeded; y: the ratio, sorted from high to low."""
    y = np.sort(np.asarray(ratio, dtype=float))[::-1]
    return np.arange(len(y)) / len(y) * 100.0, y


def period_label(times):
    first, last = times.min(), times.max()
    if (first.year, first.month) == (last.year, last.month):
        return f"{first:%b %Y}"
    return f"{first:%b %Y} to {last:%b %Y}"


def plot_headroom_analysis():
    weather = load_weather()
    df = add_dlr_columns(weather)
    period = period_label(df["time"])
    sheltering = config.SHELTERING_FACTOR
    lo_s, hi_s = min(config.SHELTERING_SWEEP), max(config.SHELTERING_SWEEP)

    # range over the plausible sheltering factors (same bearing), and the worst bearing of the sweep
    ratio_lo = add_dlr_columns(weather, sheltering=lo_s)["headroom_ratio"]
    ratio_hi = add_dlr_columns(weather, sheltering=hi_s)["headroom_ratio"]
    scenarios = scenario_table(weather, shelterings=(sheltering,), diffuse_models=(config.DIFFUSE_MODEL,))
    worst_bearing = float(worst_span(scenarios)["bearing_deg"].iloc[0])
    ratio_worst = add_dlr_columns(weather, line=replace(config.LINE, bearing_deg=worst_bearing))["headroom_ratio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=300)

    # Left: duration curve of DLR / seasonal static rating (1.0 = no gain over static)
    x, central = duration_curve(df["headroom_ratio"])
    _, lo = duration_curve(ratio_lo)
    _, hi = duration_curve(ratio_hi)
    _, worst = duration_curve(ratio_worst)
    ax1.fill_between(x, lo, hi, color="navy", alpha=0.15, linewidth=0,
                     label=f"sheltering {lo_s}-{hi_s} (bearing {config.LINE.bearing_deg:.0f} deg)")
    ax1.plot(x, central, color="navy", linewidth=2,
             label=f"central: bearing {config.LINE.bearing_deg:.0f} deg, sheltering {sheltering}")
    ax1.plot(x, worst, color="darkorange", linewidth=1.5, linestyle="--",
             label=f"worst bearing of sweep ({worst_bearing:.0f} deg), sheltering {sheltering}")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Seasonal static rating (= 1.0)")
    ax1.set_title(f"Rating duration curve, north Dublin ERA5 point, {period}", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Percentage of time ratio is exceeded (%)", fontsize=10)
    ax1.set_ylabel("DLR / seasonal static rating", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=7.5)
    below = lambda r: (np.asarray(r) < 1.0).mean() * 100.0
    ax1.text(0.02, 0.04,
             f"Hours below static rating:\ncentral {below(df['headroom_ratio']):.1f}%, "
             f"worst bearing {below(ratio_worst):.1f}%,\nsheltering {hi_s}: {below(ratio_hi):.1f}% / {lo_s}: {below(ratio_lo):.1f}%",
             transform=ax1.transAxes, fontsize=8, va="bottom",
             bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9))

    # Right: monthly headroom distribution (central case)
    df["month"] = df["time"].dt.month
    present = sorted(df["month"].unique())          # a partial year has fewer than 12 months
    monthly_data = [df.loc[df["month"] == m, "headroom_pct"].values for m in present]
    month_names = [calendar.month_abbr[m] for m in present]
    ax2.boxplot(monthly_data, tick_labels=month_names, patch_artist=True,
                boxprops=dict(facecolor="skyblue", color="darkblue"),
                medianprops=dict(color="darkred", linewidth=1.5))
    pooled = " (years pooled)" if df["time"].dt.year.nunique() > 1 else ""
    ax2.set_title("Monthly headroom over static rating, central case", fontsize=10, fontweight="bold")
    ax2.set_xlabel(f"Month{pooled}", fontsize=10)
    ax2.set_ylabel("Headroom over static rating (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    static = ", ".join(f"{k} {v:.0f} A" for k, v in df.attrs["static_ratings"].items())
    fig.text(0.01, 0.005,
             f"Drake ACSR, {config.DRAKE.t_max_c:.0f} degC limit. Static ratings are ILLUSTRATIVE assumptions ({static}). "
             f"ERA5 hourly wind/temperature/sun at one grid point; wind shear and sheltering are assumed, not measured.",
             fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir = ROOT_DIR / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / OUTPUT_NAME
    fig.savefig(out_path)
    print(f"Saved headroom figures ({period}) to {out_path}")

if __name__ == "__main__":
    plot_headroom_analysis()
