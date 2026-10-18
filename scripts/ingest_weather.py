"""
Fetch hourly ERA5 reanalysis weather for the North Dublin corridor via Open-Meteo.

Units written to the CSV: t_ambient degC, v_wind m/s (10 m), wind_dir deg (from),
solar_* W/m2 (mean of the preceding hour).
"""

from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent

# Reference point for the North Dublin 220 kV corridor (illustrative, not a surveyed span)
LAT, LON = 53.45, -6.22
START_DATE, END_DATE = "2023-01-01", "2023-12-31"

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "direct_radiation",
    "diffuse_radiation",
]


def fetch_irish_corridor_weather():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",  # API default is km/h, which silently broke the DLR maths
        "models": "era5",         # default "best_match" blends several models
        "timezone": "UTC",
    }

    print(f"Fetching ERA5 for ({LAT}, {LON}), {START_DATE} to {END_DATE}...")
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if payload["hourly_units"]["wind_speed_10m"] != "m/s":
        raise ValueError(f"Unexpected wind unit: {payload['hourly_units']['wind_speed_10m']}")

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "t_ambient": hourly["temperature_2m"],
        "v_wind": hourly["wind_speed_10m"],
        "wind_dir": hourly["wind_direction_10m"],
        "solar_direct": hourly["direct_radiation"],
        "solar_diffuse": hourly["diffuse_radiation"],
    })

    n_before = len(df)
    df = df.dropna()
    if len(df) != n_before:
        print(f"Warning: dropped {n_before - len(df)} rows with missing values")

    # Global horizontal irradiance = direct (horizontal) + diffuse
    df["solar_rad"] = df["solar_direct"] + df["solar_diffuse"]

    # Plausibility guard: 10 m annual-mean wind in Dublin is ~4-6 m/s
    mean_v = df["v_wind"].mean()
    if not 2.0 <= mean_v <= 9.0:
        raise ValueError(f"Mean wind {mean_v:.1f} m/s is implausible; check units")

    out_dir = ROOT_DIR / "data" / "raw_weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "north_dublin_2023_hourly.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path} (mean wind {mean_v:.2f} m/s)")


if __name__ == "__main__":
    fetch_irish_corridor_weather()
    