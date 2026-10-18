"""
Plot annual DLR ampacity duration curve and monthly headroom distributions
for the North Dublin 220 kV corridor.
"""

import numpy as np
import matplotlib.pyplot as plt

from dlr.pipeline import ROOT_DIR, add_dlr_columns, load_weather


def plot_headroom_analysis():
    df = add_dlr_columns(load_weather())
    static_rating = df.attrs["static_rating"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # 1. Ampacity Duration Curve
    sorted_dlr = np.sort(df["dlr_ampacity"].values)[::-1]
    hours_axis = np.arange(len(sorted_dlr)) / 8760.0 * 100.0 # percentage of year
    
    ax1.plot(hours_axis, sorted_dlr, color="navy", linewidth=2, label="Dynamic Rating (DLR)")
    ax1.axhline(static_rating, color="red", linestyle="--", linewidth=1.5, label=f"Static Rating ({static_rating:.0f} A)")
    ax1.set_title("Annual Ampacity Duration Curve (North Dublin 220 kV)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Percentage of Time (%)", fontsize=10)
    ax1.set_ylabel("Ampacity (A)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # 2. Monthly Headroom Distribution Boxplot
    df["month"] = df["time"].dt.month
    monthly_data = [group["headroom_pct"].values for _, group in df.groupby("month")]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
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
