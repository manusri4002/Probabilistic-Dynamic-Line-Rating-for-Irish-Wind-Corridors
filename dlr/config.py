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
    bearing_deg: float = 45.0        # ILLUSTRATIVE: one assumed bearing; see worst-span analysis
    conductor_height_m: float = 25.0
    elevation_m: float = 0.0
    latitude: float = 53.45          # North Dublin reference point, drives the sun position
    longitude: float = -6.22


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

WEATHER_CSV = "data/raw_weather/north_dublin_hourly.csv"
WEATHER_START = "2026-01-01"
WEATHER_END = "2026-09-30"
MET_HEIGHT_M = 10.0      
SHEAR_ALPHA = 0.143    
SHELTERING_FACTOR = 0.7
SHELTERING_SWEEP = (1.0, 0.8, 0.7, 0.6, 0.4)  
WIND_FLOOR_MS = 0.1

# Solar heat gain. ASSUMPTIONS: ground albedo is site-dependent (grass/farmland roughly 0.1-0.25).
GROUND_ALBEDO = 0.2
DIFFUSE_MODEL = "cigre601"       # or "isotropic" (CIGRE TB 207 form); the pair is a modelling band
DIFFUSE_MODELS = ("cigre601", "isotropic")

# Bands (dlr/bands.py). A line axis at b and b + 180 degrees is the same line, so 0-165 covers every line.
# The real bearing of a span is unknown here, so results are reported for the worst bearing in this sweep.
BEARING_SWEEP = tuple(range(0, 180, 15))
SOLAR_TIME_OFFSET_MIN = -30      # Open-Meteo radiation is the mean of the PRECEDING hour: evaluate the sun mid-hour

#Day-ahead forecast
FORECAST_ISSUE_HOUR_UTC = 12   # forecast for day D is issued at this hour on day D-1 (lead 12-35 h)
FORECAST_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
FORECAST_RISK_QUANTILES = (0.05, 0.10)   # low quantiles: the "how little headroom can I count on" numbers
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)        # train / calibration / test, assigned in whole blocks
SPLIT_BLOCK_DAYS = 7
SPLIT_GAP_DAYS = 2             # days dropped at the edge of a block whose neighbour is in another set
SPLIT_SEED = 0
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                   random_state=42, verbose=-1)