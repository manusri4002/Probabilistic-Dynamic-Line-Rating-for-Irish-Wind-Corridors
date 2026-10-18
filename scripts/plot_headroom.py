"""
Plot annual DLR ampacity duration curve and monthly headroom distributions
for the North Dublin 220 kV corridor.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "physics"))
from ieee738_v0 import calculate_ampacity

def plot_headroom_analysis():
    weather_path = ROOT_DIR / "data" / "raw_weather" / "north_dublin_2023_hourly.csv"
    df = pd.read_csv(weather_path)
    df["time"] = pd.to_datetime(df["time"])
    
    d_m = 0.02814
    t_c = 75.0
    r_75 = 8.688e-5
    line_bearing_deg = 45.0
    
    # Static baseline rating
    static_rating = calculate_ampacity(t_c, 25.0, 0.5, 90.0, 1000.0, d_m, r_75)
    
    # Compute hourly DLR
    dlr_list = []
    for _, row in df.iterrows():
        wind_dir = row["wind_dir"]
        phi_deg = abs(wind_dir - line_bearing_deg) % 180.0
        if phi_deg > 90.0:
            phi_deg = 180.0 - phi_deg
            
        i_dlr = calculate_ampacity(
            t_c=t_c,
            t_a=row["t_ambient"],
            v_wind=max(row["v_wind"], 0.1),
            phi_deg=phi_deg,
            solar_rad=row["solar_rad"],
            d_m=d_m,
            r_tc=r_75
        )
        dlr_list.append(i_dlr)
        
    df["dlr_ampacity"] = dlr_list
    df["headroom_pct"] = ((df["dlr_ampacity"] - static_rating) / static_rating) * 100.0
    
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
