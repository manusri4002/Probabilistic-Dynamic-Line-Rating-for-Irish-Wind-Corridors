"""Single source of truth for conductor, line and baseline assumptions.

Change a number here and every script follows. Values marked ILLUSTRATIVE are
assumptions, not surveyed data: replace them with the real line's data.
"""

from dataclasses import dataclass
@dataclass(frozen=True)
class Conductor:
    name: str
    diameter_m: float      
    r_at_tmax: float       
    t_max_c: float         
    emissivity: float = 0.5
    absorptivity: float = 0.5


@dataclass(frozen=True)
class Line:
    bearing_deg: float = 45.0        
    conductor_height_m: float = 25.0  
    elevation_m: float = 0.0          


@dataclass(frozen=True)
class StaticCase:
    """Fixed weather assumptions behind the static (planning) rating.
    ILLUSTRATIVE: replace with the operator's published seasonal assumptions."""
    t_a: float = 25.0
    v_wind: float = 0.5
    phi_deg: float = 90.0
    solar_rad: float = 1000.0

# Seasonal static (planning) ratings
SEASONAL_STATIC = {
    "winter": StaticCase(t_a=5.0),
    "spring_autumn": StaticCase(t_a=12.0),
    "summer": StaticCase(t_a=25.0),
}
SEASON_BY_MONTH = {
    11: "winter", 12: "winter", 1: "winter", 2: "winter",
    3: "spring_autumn", 4: "spring_autumn", 9: "spring_autumn", 10: "spring_autumn",
    5: "summer", 6: "summer", 7: "summer", 8: "summer",
}
SEASONS = ("winter", "spring_autumn", "summer")
DRAKE = Conductor("Drake ACSR 26/7", diameter_m=0.02814, r_at_tmax=8.688e-5, t_max_c=75.0)
LINE = Line()
STATIC = SEASONAL_STATIC["summer"]

WEATHER_CSV = "data/raw_weather/north_dublin_2023_hourly.csv"
MET_HEIGHT_M = 10.0      
SHEAR_ALPHA = 0.143    
SHELTERING_FACTOR = 0.7
SHELTERING_SWEEP = (1.0, 0.8, 0.7, 0.6, 0.4)  
WIND_FLOOR_MS = 0.1      
