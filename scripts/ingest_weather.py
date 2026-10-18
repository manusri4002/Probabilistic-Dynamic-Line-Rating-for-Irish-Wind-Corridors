"""
Fetch hourly ERA5 reanalysis weather data via Open-Meteo for North Dublin corridor.
Outputs a clean CSV ready to ingest into the IEEE 738 thermal solver.
"""

import requests
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def fetch_irish_corridor_weather():
    # Coordinates for North Dublin 220 kV corridor reference point
    lat, lon = 53.45, -6.22
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,direct_radiation,diffuse_radiation",
        "timezone": "UTC"
    }
    
    print(f"Fetching ERA5 reanalysis archive for North Dublin ({lat}, {lon})...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "t_ambient": data["hourly"]["temperature_2m"],
        "v_wind": data["hourly"]["wind_speed_10m"],
        "wind_dir": data["hourly"]["wind_direction_10m"],
        "solar_direct": data["hourly"]["direct_radiation"],
        "solar_diffuse": data["hourly"]["diffuse_radiation"]
    })
    
    # Total global solar radiation approximation on horizontal/inclined surface
    df["solar_rad"] = df["solar_direct"].fillna(0) + df["solar_diffuse"].fillna(0)
    
    # Clean up
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna()
    
    out_dir = ROOT_DIR / "data" / "raw_weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "north_dublin_2023_hourly.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved weather time series to {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    fetch_irish_corridor_weather()
    