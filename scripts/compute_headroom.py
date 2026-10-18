"""
Compute hourly DLR ampacity versus static baseline rating for the North Dublin corridor.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "physics"))
from ieee738_v0 import calculate_ampacity

def run_headroom_analysis():
    weather_path = ROOT_DIR / "data" / "raw_weather" / "north_dublin_2023_hourly.csv"
    if not weather_path.exists():
        print(f"Weather data not found at {weather_path}. Run ingest_weather.py first.")
        return

    df = pd.read_csv(weather_path)
    
    # Conductor & line parameters
    d_m = 0.02814        # Drake ACSR diameter
    t_c = 75.0           # Max operating temp
    r_75 = 8.688e-5      # Resistance at 75C
    line_bearing_deg = 45.0  # Assumed orientation of the 220 kV corridor
    
    # Static rating baseline (conservative EirGrid style assumptions)
    static_t_a = 25.0
    static_v = 0.5
    static_phi = 90.0
    static_solar = 1000.0
    static_rating = calculate_ampacity(t_c, static_t_a, static_v, static_phi, static_solar, d_m, r_75)
    print(f"Baseline Static Rating: {static_rating:.1f} A")
    
    # Compute hourly DLR
    dlr_list = []
    for _, row in df.iterrows():
        # Relative wind angle calculation
        wind_dir = row["wind_dir"]
        phi_deg = abs(wind_dir - line_bearing_deg) % 180.0
        if phi_deg > 90.0:
            phi_deg = 180.0 - phi_deg
            
        i_dlr = calculate_ampacity(
            t_c=t_c,
            t_a=row["t_ambient"],
            v_wind=max(row["v_wind"], 0.1), # floor wind speed to avoid 0 div/NaN issues
            phi_deg=phi_deg,
            solar_rad=row["solar_rad"],
            d_m=d_m,
            r_tc=r_75
        )
        dlr_list.append(i_dlr)
        
    df["dlr_ampacity"] = dlr_list
    df["headroom_pct"] = ((df["dlr_ampacity"] - static_rating) / static_rating) * 100.0
    
    # Summary stats
    print(f"Mean DLR Ampacity: {df['dlr_ampacity'].mean():.1f} A")
    print(f"Median Headroom Increase over Static: {df['headroom_pct'].median():.1f}%")
    print(f"Percentage of time above static rating: {(df['dlr_ampacity'] > static_rating).mean() * 100:.1f}%")

if __name__ == "__main__":
    run_headroom_analysis()
    