"""Turn a weather table into DLR ampacity. All scripts call this; none re-implement it."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .ieee738 import calculate_ampacity
from .solar import dni_from_direct_horizontal, effective_irradiance, sun_position

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


def wind_at_conductor(v_met, line=config.LINE, apply_shear=True, sheltering=1.0):
    """Reanalysis 10 m wind -> effective wind at the conductor.

    1. power-law shear up to conductor height,
    2. sheltering factor (<= 1) for terrain/obstacles the grid box cannot see,
    3. a small floor so the forced-convection term stays defined in calm air."""
    v = np.asarray(v_met, dtype=float)
    if apply_shear:
        v = v * (line.conductor_height_m / config.MET_HEIGHT_M) ** config.SHEAR_ALPHA
    return np.maximum(v * sheltering, config.WIND_FLOOR_MS)


def solar_on_conductor(df, line=config.LINE, solar_model="auto",
                       albedo=config.GROUND_ALBEDO, diffuse_model=config.DIFFUSE_MODEL):
    """Irradiance (W/m2) on the conductor's projected area for every row of df.

    solar_model="geometric": sun position from time and the line's latitude/longitude, direct beam
        resolved against the line bearing, diffuse and ground-reflected terms added (dlr/solar.py).
        Needs `solar_diffuse` plus either `solar_dni` or `solar_direct` (horizontal) columns.
    solar_model="ghi": the old shortcut, global horizontal irradiance used as if it fell on the
        projected area. Kept only for comparison; it ignores sun angle and line bearing.
    "auto" picks geometric and refuses to fall back silently when the columns are missing."""
    if solar_model not in ("auto", "geometric", "ghi"):
        raise ValueError("solar_model must be 'auto', 'geometric' or 'ghi'")
    if solar_model == "ghi":
        return df["solar_rad"].to_numpy(dtype=float)

    cols = set(df.columns)
    if "solar_diffuse" not in cols or not ({"solar_dni", "solar_direct"} & cols):
        raise ValueError("geometric solar model needs 'solar_diffuse' and 'solar_dni' (or 'solar_direct'); "
                         "pass solar_model='ghi' to use the old horizontal shortcut deliberately")
    mid_hour = df["time"] + pd.Timedelta(minutes=config.SOLAR_TIME_OFFSET_MIN)
    elev, azim = sun_position(mid_hour, line.latitude, line.longitude)
    if "solar_dni" in cols:
        dni = df["solar_dni"].to_numpy(dtype=float)
    else:
        dni = dni_from_direct_horizontal(df["solar_direct"].to_numpy(dtype=float), elev)
    return effective_irradiance(dni, df["solar_diffuse"].to_numpy(dtype=float), elev, azim,
                                line.bearing_deg, albedo, diffuse_model)


def static_rating(conductor=config.DRAKE, case=config.STATIC, line=config.LINE):
    return calculate_ampacity(
        conductor.t_max_c, case.t_a, case.v_wind, case.phi_deg, case.solar_rad,
        conductor.diameter_m, conductor.r_at_tmax,
        conductor.emissivity, conductor.absorptivity, line.elevation_m,
    )

def seasonal_static_ratings(conductor=config.DRAKE, line=config.LINE, cases=config.SEASONAL_STATIC):
    """{season: static rating in A}"""
    return {name: static_rating(conductor, case, line) for name, case in cases.items()}

def compute_dlr(df, conductor=config.DRAKE, line=config.LINE, apply_shear=True,
                sheltering=config.SHELTERING_FACTOR, solar_model="auto"):
    """Hourly ampacity (A). The physics stays scalar (portable to MicroPython);
    we iterate over plain arrays, which is far faster than DataFrame.iterrows()."""
    v = wind_at_conductor(df["v_wind"].to_numpy(), line, apply_shear, sheltering)
    phi = attack_angle_deg(df["wind_dir"].to_numpy(), line.bearing_deg)
    t_a = df["t_ambient"].to_numpy()
    sol = solar_on_conductor(df, line, solar_model)
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
    """Return a copy of df with season, static_rating, dlr_ampacity, headroom_ratio and
    headroom_pct. Headroom is measured against the static rating of that row's season."""
    if "time" not in df.columns:
        raise ValueError("df needs a 'time' column to assign each hour to a season")
    out = df.copy()
    ratings = seasonal_static_ratings(kwargs.get("conductor", config.DRAKE), kwargs.get("line", config.LINE))
    out["season"] = out["time"].dt.month.map(config.SEASON_BY_MONTH)
    out["static_rating"] = out["season"].map(ratings)
    out["dlr_ampacity"] = compute_dlr(out, **kwargs)
    out["headroom_ratio"] = out["dlr_ampacity"] / out["static_rating"]
    out["headroom_pct"] = (out["headroom_ratio"] - 1.0) * 100.0
    out.attrs["static_ratings"] = ratings
    return out


def headroom_summary(df):
    """Per-season and overall headroom statistics from add_dlr_columns output.

    P5 / P1 headroom: the headroom that is exceeded 95% / 99% of the time (the number an
    operator can rely on). below_static_pct: share of hours when DLR is *lower* than static."""
    def stats(g):
        r = g["headroom_ratio"]
        return pd.Series({
            "static_A": g["static_rating"].iloc[0] if g["static_rating"].nunique() == 1 else np.nan,
            "median_%": (r.median() - 1) * 100,
            "P5_%": (r.quantile(0.05) - 1) * 100,
            "P1_%": (r.quantile(0.01) - 1) * 100,
            "below_static_%": (r < 1.0).mean() * 100,
        })
    rows = {s: stats(df[df["season"] == s]) for s in config.SEASONS if (df["season"] == s).any()}
    rows["all year"] = stats(df)
    return pd.DataFrame(rows).T
