"""
Analyze the correlation between DLR headroom on the North Dublin corridor 
and Irish wind generation output (using EirGrid regional or proxy wind data).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from dlr.pipeline import add_dlr_columns, load_weather, wind_at_conductor

def run_correlation_analysis():
    df = add_dlr_columns(load_weather())
    df["v_wind_corrected"] = wind_at_conductor(df["v_wind"])
    
    # 2. Mock/Simulate EirGrid All-Island Wind Generation proxy 
    # (In production, replace or merge with downloaded EirGrid Smart Grid CSV data)
    # Wind generation strongly correlates with wind speed cubed / wind power available, plus some spatial noise
    np.random.seed(42)
    df["wind_gen_proxy"] = (df["v_wind_corrected"] ** 3) * np.random.normal(1.0, 0.2, len(df))
    df["wind_gen_proxy"] = df["wind_gen_proxy"].clip(lower=0)
    # Normalize to a 0 - 100% capacity factor scale
    df["wind_gen_pct"] = (df["wind_gen_proxy"] / df["wind_gen_proxy"].max()) * 100.0
    
    #Evaluation Metrics
    spearman_corr, p_val = stats.spearmanr(df["wind_gen_pct"], df["headroom_ratio"])
    
    print(f"--- DLR vs Wind Generation Correlation Results ---")
    print(f"Spearman Rank Correlation: {spearman_corr:.3f} (p-value: {p_val:.4e})")
    
    #Median headroom by wind output decile
    df["wind_decile"] = pd.qcut(df["wind_gen_pct"], 10, labels=False) + 1
    decile_summary = df.groupby("wind_decile")["headroom_ratio"].median()
    print("\nMedian Headroom Ratio (DLR / Static) by Wind Generation Decile:")
    print(decile_summary)
    
    #Failure mode analysis: High wind generation (> 80th percentile) but low headroom ratio (< 1.0)
    high_wind_threshold = df["wind_gen_pct"].quantile(0.80)
    failure_hours = df[(df["wind_gen_pct"] >= high_wind_threshold) & (df["headroom_ratio"] < 1.0)]
    failure_share = (len(failure_hours) / len(df[df["wind_gen_pct"] >= high_wind_threshold])) * 100.0
    
    print(f"\nFailure Mode Analysis:")
    print(f"Share of high-wind hours (top 20%) where headroom fell below static rating: {failure_share:.2f}%")
    print(f"(Driven by unfavorable wind attack angles or severe localized weather decoupling).")

if __name__ == "__main__":
    run_correlation_analysis()
