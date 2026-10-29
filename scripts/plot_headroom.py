"""
Plot annual DLR ampacity duration curve and monthly headroom distributions
for the North Dublin 220 kV corridor.
"""

import numpy as np
import matplotlib.pyplot as plt
import calendar
from dlr.pipeline import ROOT_DIR, add_dlr_columns, load_weather


def plot_headroom_analysis():
    df = add_dlr_columns(load_weather())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Duration curve of DLR / seasonal static rating (1.0 = no gain over static)
    sorted_ratio = np.sort(df["headroom_ratio"].values)[::-1]
    hours_axis = np.arange(len(sorted_ratio)) / len(sorted_ratio) * 100.0  # percentage of year

    ax1.plot(hours_axis, sorted_ratio, color="navy", linewidth=2, label="DLR / seasonal static rating")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Static rating (= 1.0)")
    ax1.set_title("Rating Duration Curve (North Dublin 220 kV, illustrative)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Percentage of Time Ratio Is Exceeded (%)", fontsize=10)
    ax1.set_ylabel("DLR / static rating", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # 2. Monthly Headroom Distribution Boxplot
    df["month"] = df["time"].dt.month
    present = sorted(df["month"].unique())          # a partial year has fewer than 12 months
    monthly_data = [df.loc[df["month"] == m, "headroom_pct"].values for m in present]
    month_names = [calendar.month_abbr[m] for m in present]
    
    ax2.boxplot(monthly_data, tick_labels=month_names, patch_artist=True,
                boxprops=dict(facecolor="skyblue", color="darkblue"),
                medianprops=dict(color="darkred", linewidth=1.5))
    ax2.set_title("Monthly Headroom Increase Over Static Rating", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Month", fontsize=10)
    ax2.set_ylabel("Headroom Increase (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    fig.tight_layout()
    out_dir = ROOT_DIR / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dlr_annual_headroom_analysis.png"
    fig.savefig(out_path)
    print(f"Saved annual headroom figures to {out_path}")

if __name__ == "__main__":
    plot_headroom_analysis()
