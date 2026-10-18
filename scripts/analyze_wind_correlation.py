"""
Analyze the correlation between DLR headroom on the North Dublin corridor 
and Irish wind generation output (using EirGrid regional or proxy wind data).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "physics"))
from ieee738_v0 import calculate_ampacity

def run_correlation_analysis():
    weather_path = ROOT_DIR / "data" / "raw_weather" / "north_dublin_2023_hourly.csv"
    if not weather_path.exists():
        print(f"Weather data not found at {weather_path}.")
        return

    df = pd.read_csv(weather_path)
    df["time"] = pd.to_datetime(df["time"])
    
    # 1. Wind Shear Correction (10m to 25m conductor height)
    h_10 = 10.0
    h_con = 25.0
    alpha = 0.143
    df["v_wind_corrected"] = df["v_wind"] * ((h_con / h_10) ** alpha)
    
    # Conductor specs & Line orientation
    d_m = 0.02814
    t_c = 75.0
    r_75 = 8.688e-5
    line_bearing_deg = 45.0 # North Dublin corridor orientation approximation
    
    # Static rating baseline
    static_rating = calculate_ampacity(t_c, 25.0, 0.5, 90.0, 1000.0, d_m, r_75)
    
    # Compute hourly DLR with corrected wind and dynamic attack angle
    dlr_list = []
    for _, row in df.iterrows():
        wind_dir = row["wind_dir"]
        phi_deg = abs(wind_dir - line_bearing_deg) % 180.0
        if phi_deg > 90.0:
            phi_deg = 180.0 - phi_deg
            
        i_dlr = calculate_ampacity(
            t_c=t_c,
            t_a=row["t_ambient"],
            v_wind=max(row["v_wind_corrected"], 0.1),
            phi_deg=phi_deg,
            solar_rad=row["solar_rad"],
            d_m=d_m,
            r_tc=r_75
        )
        dlr_list.append(i_dlr)
        
    df["dlr_ampacity"] = dlr_list
    df["headroom_ratio"] = df["dlr_ampacity"] / static_rating
    
    # 2. Mock/Simulate EirGrid All-Island Wind Generation proxy 
    # (In production, replace or merge with downloaded EirGrid Smart Grid CSV data)
    # Wind generation strongly correlates with wind speed cubed / wind power available, plus some spatial noise
    np.random.seed(42)
    df["wind_gen_proxy"] = (df["v_wind_corrected"] ** 3) * np.random.normal(1.0, 0.2, len(df))
    df["wind_gen_proxy"] = df["wind_gen_proxy"].clip(lower=0)
    # Normalize to a 0 - 100% capacity factor scale
    df["wind_gen_pct"] = (df["wind_gen_proxy"] / df["wind_gen_proxy"].max()) * 100.0
    
    # 3. Evaluation Metrics
    spearman_corr, p_val = stats.spearmanr(df["wind_gen_pct"], df["headroom_ratio"])
    
    print(f"--- DLR vs Wind Generation Correlation Results ---")
    print(f"Spearman Rank Correlation: {spearman_corr:.3f} (p-value: {p_val:.4e})")
    
    # Median headroom by wind output decile
    df["wind_decile"] = pd.qcut(df["wind_gen_pct"], 10, labels=False) + 1
    decile_summary = df.groupby("wind_decile")["headroom_ratio"].median()
    print("\nMedian Headroom Ratio (DLR / Static) by Wind Generation Decile:")
    print(decile_summary)
    
    # Failure mode analysis: High wind generation (> 80th percentile) but low headroom ratio (< 1.0)
    high_wind_threshold = df["wind_gen_pct"].quantile(0.80)
    failure_hours = df[(df["wind_gen_pct"] >= high_wind_threshold) & (df["headroom_ratio"] < 1.0)]
    failure_share = (len(failure_hours) / len(df[df["wind_gen_pct"] >= high_wind_threshold])) * 100.0
    
    print(f"\nFailure Mode Analysis:")
    print(f"Share of high-wind hours (top 20%) where headroom fell below static rating: {failure_share:.2f}%")
    print(f"(Driven by unfavorable wind attack angles or severe localized weather decoupling).")

if __name__ == "__main__":
    run_correlation_analysis()
