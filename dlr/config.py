"""Single source of truth for conductor, line and baseline assumptions.

Change a number here and every script follows. Values marked ILLUSTRATIVE are
assumptions, not surveyed data: replace them with the real line's data.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Conductor:
    name: str
    diameter_m: float      # overall diameter (m)
    r_at_tmax: float       # AC resistance at t_max_c (ohm/m)
    t_max_c: float         # maximum allowable conductor temperature (degC)
    emissivity: float = 0.5
    absorptivity: float = 0.5


@dataclass(frozen=True)
class Line:
    bearing_deg: float = 45.0        # ILLUSTRATIVE: compass bearing of the span axis
    conductor_height_m: float = 25.0  # ILLUSTRATIVE: height above ground
    elevation_m: float = 0.0          # ILLUSTRATIVE: height above sea level


@dataclass(frozen=True)
class StaticCase:
    """Fixed weather assumptions behind the static (planning) rating.
    ILLUSTRATIVE: replace with the operator's published seasonal assumptions."""
    t_a: float = 25.0
    v_wind: float = 0.5
    phi_deg: float = 90.0
    solar_rad: float = 1000.0


DRAKE = Conductor("Drake ACSR 26/7", diameter_m=0.02814, r_at_tmax=8.688e-5, t_max_c=75.0)
LINE = Line()
STATIC = StaticCase()

WEATHER_CSV = "data/raw_weather/north_dublin_2023_hourly.csv"
MET_HEIGHT_M = 10.0      # height of the reanalysis wind
SHEAR_ALPHA = 0.143      # power-law exponent (open terrain)
WIND_FLOOR_MS = 0.1      # keeps the forced-convection term defined in calm air
