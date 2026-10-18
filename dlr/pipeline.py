"""Turn a weather table into DLR ampacity. All scripts call this; none re-implement it."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .ieee738 import calculate_ampacity

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_weather(path=None):
    """Read the weather CSV and refuse implausible data (e.g. wrong wind units)."""
    path = Path(path) if path else ROOT_DIR / config.WEATHER_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/ingest_weather.py first.")
    df = pd.read_csv(path, parse_dates=["time"])
    mean_v = df["v_wind"].mean()
    if not 2.0 <= mean_v <= 9.0:
        raise ValueError(f"Mean wind {mean_v:.1f} m/s is implausible; are the units m/s?")
    return df


def attack_angle_deg(wind_dir_deg, bearing_deg):
    """Angle between wind and conductor axis, folded into 0-90 degrees.

    Wind direction is where the wind comes FROM, but only the axis matters, so
    we fold modulo 180 and then reflect anything above 90."""
    d = np.abs(np.asarray(wind_dir_deg, dtype=float) - bearing_deg) % 180.0
    return np.where(d > 90.0, 180.0 - d, d)


def wind_at_conductor(v_met, line=config.LINE, apply_shear=True):
    """Scale reanalysis wind (10 m) to conductor height with a power law."""
    v = np.asarray(v_met, dtype=float)
    if apply_shear:
        v = v * (line.conductor_height_m / config.MET_HEIGHT_M) ** config.SHEAR_ALPHA
    return np.maximum(v, config.WIND_FLOOR_MS)


def static_rating(conductor=config.DRAKE, case=config.STATIC, line=config.LINE):
    return calculate_ampacity(
        conductor.t_max_c, case.t_a, case.v_wind, case.phi_deg, case.solar_rad,
        conductor.diameter_m, conductor.r_at_tmax,
        conductor.emissivity, conductor.absorptivity, line.elevation_m,
    )


def compute_dlr(df, conductor=config.DRAKE, line=config.LINE, apply_shear=True):
    """Hourly ampacity (A). The physics stays scalar (portable to MicroPython);
    we iterate over plain arrays, which is far faster than DataFrame.iterrows()."""
    v = wind_at_conductor(df["v_wind"].to_numpy(), line, apply_shear)
    phi = attack_angle_deg(df["wind_dir"].to_numpy(), line.bearing_deg)
    t_a = df["t_ambient"].to_numpy()
    sol = df["solar_rad"].to_numpy()
    out = [
        calculate_ampacity(
            conductor.t_max_c, ta, vi, pi, si,
            conductor.diameter_m, conductor.r_at_tmax,
            conductor.emissivity, conductor.absorptivity, line.elevation_m,
        )
        for ta, vi, pi, si in zip(t_a, v, phi, sol)
    ]
    return pd.Series(out, index=df.index, name="dlr_ampacity")


def add_dlr_columns(df, **kwargs):
    """Return a copy of df with dlr_ampacity, headroom_ratio and headroom_pct added."""
    out = df.copy()
    rating = static_rating(kwargs.get("conductor", config.DRAKE), line=kwargs.get("line", config.LINE))
    out["dlr_ampacity"] = compute_dlr(out, **kwargs)
    out["headroom_ratio"] = out["dlr_ampacity"] / rating
    out["headroom_pct"] = (out["headroom_ratio"] - 1.0) * 100.0
    out.attrs["static_rating"] = rating
    return out