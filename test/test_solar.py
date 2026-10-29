import numpy as np
import pandas as pd
import pytest

from dlr import config
from dlr.pipeline import add_dlr_columns, compute_dlr, solar_on_conductor
from dlr.solar import (dni_from_direct_horizontal, effective_irradiance, sin_incidence_angle,
                       sun_position)

LAT, LON = 53.45, -6.22

# Reference values computed once with pvlib 0.16.1 (geometric, no refraction) for (LAT, LON).
PVLIB_REFERENCE = [
    ("2025-06-21 12:25", 59.98, 179.20),
    ("2025-12-21 12:20", 13.11, 179.28),
    ("2025-03-20 09:00", 20.96, 121.13),
    ("2025-09-22 16:00", 19.67, 241.24),
    ("2025-12-21 00:00", -59.68, 349.59),
]


@pytest.mark.parametrize("when, elevation, azimuth", PVLIB_REFERENCE)
def test_sun_position_matches_pvlib_reference(when, elevation, azimuth):
    el, az = sun_position(pd.DatetimeIndex([when]), LAT, LON)
    assert el[0] == pytest.approx(elevation, abs=0.6)
    assert (az[0] - azimuth + 180) % 360 - 180 == pytest.approx(0.0, abs=0.6)


def test_solstice_noon_elevations_follow_90_minus_lat_plus_declination():
    summer, _ = sun_position(pd.DatetimeIndex(["2025-06-21 12:25"]), LAT, LON)
    winter, _ = sun_position(pd.DatetimeIndex(["2025-12-21 12:20"]), LAT, LON)
    assert summer[0] == pytest.approx(90 - LAT + 23.44, abs=0.6)
    assert winter[0] == pytest.approx(90 - LAT - 23.44, abs=0.6)


@pytest.mark.parametrize("elev, azim, line, expected", [
    (30.0, 180.0, 90.0, 1.0),     # sun due south, east-west line: rays perpendicular to the axis
    (30.0, 180.0, 0.0, 0.5),      # north-south line: only the sun's elevation lights the diameter
    (90.0, 123.0, 17.0, 1.0),     # sun overhead: perpendicular to any horizontal line
    (0.0, 90.0, 90.0, 0.0),       # sun on the horizon, straight along the line
    (30.0, 0.0, 0.0, 0.5),        # line azimuth vs sun azimuth differing by 180 changes nothing
])
def test_incidence_angle_geometry(elev, azim, line, expected):
    assert sin_incidence_angle(elev, azim, line) == pytest.approx(expected, abs=1e-12)


def test_line_azimuth_and_its_reverse_are_equivalent():
    a = sin_incidence_angle(25.0, 200.0, 45.0)
    b = sin_incidence_angle(25.0, 200.0, 225.0)
    assert a == pytest.approx(b)


def test_isotropic_diffuse_factor_is_pi_over_two_by_integration():
    """A horizontal cylinder (diameter D) under uniform sky light absorbs (pi/2) * D * I_d per metre.
    Integrate the sky view factor (1 + cos(beta)) / 2 around the circumference to check."""
    d, i_d = 0.03, 100.0
    beta = np.linspace(0.0, 2.0 * np.pi, 200001)
    flux = i_d * (1.0 + np.cos(beta)) / 2.0 * (d / 2.0)
    per_metre = float(np.sum((flux[1:] + flux[:-1]) / 2.0 * np.diff(beta)))      # trapezoid rule
    assert per_metre == pytest.approx(0.5 * np.pi * d * i_d, rel=1e-6)
    got = effective_irradiance(0.0, i_d, 30.0, 180.0, 90.0, albedo=0.0, diffuse_model="isotropic")
    assert got * d == pytest.approx(per_metre, rel=1e-6)


def test_cigre601_and_isotropic_differ_only_in_the_diffuse_term():
    args = dict(elevation_deg=30.0, azimuth_deg=180.0, line_azimuth_deg=90.0, albedo=0.2)
    no_diffuse_a = effective_irradiance(500.0, 0.0, diffuse_model="cigre601", **args)
    no_diffuse_b = effective_irradiance(500.0, 0.0, diffuse_model="isotropic", **args)
    assert no_diffuse_a == pytest.approx(no_diffuse_b)
    assert effective_irradiance(0.0, 100.0, diffuse_model="isotropic", **args) > \
        effective_irradiance(0.0, 100.0, diffuse_model="cigre601", **args)


def test_unknown_diffuse_model_is_rejected():
    with pytest.raises(ValueError):
        effective_irradiance(0.0, 0.0, 30.0, 180.0, 90.0, diffuse_model="nonsense")


def test_no_beam_heating_when_the_sun_is_down():
    out = effective_irradiance(800.0, 0.0, -10.0, 0.0, 45.0)
    assert out == 0.0


def test_low_winter_sun_heats_a_perpendicular_line_far_more_than_horizontal_irradiance_suggests():
    """The defect this module fixes: at 13 degrees elevation horizontal direct irradiance is only
    sin(13 deg) of the beam, but an east-west line is lit across its full diameter."""
    dni, elev = 500.0, 13.0
    direct_horizontal = dni * np.sin(np.deg2rad(elev))
    beam_on_line = effective_irradiance(dni, 0.0, elev, 180.0, 90.0, albedo=0.0)
    assert beam_on_line == pytest.approx(dni)
    assert beam_on_line / direct_horizontal == pytest.approx(1.0 / np.sin(np.deg2rad(elev)))
    assert beam_on_line > 4.0 * direct_horizontal


def test_dni_from_direct_horizontal_round_trip_floor_and_ceiling():
    elev = np.array([60.0, 30.0, 1.0, -5.0])
    true_dni = np.array([700.0, 500.0, 100.0, 0.0])
    horizontal = true_dni * np.sin(np.deg2rad(elev))
    got = dni_from_direct_horizontal(horizontal, elev)
    assert got[:2] == pytest.approx(true_dni[:2])
    assert got[2] <= true_dni[2]            # sin(H) floor: under-estimates, never explodes
    assert got[3] == 0.0                    # sun down
    assert dni_from_direct_horizontal(500.0, 5.0) <= 1200.0


# ---- pipeline integration ---------------------------------------------------------------------

def midday(date, **kw):
    base = {"t_ambient": 10.0, "v_wind": 4.0, "wind_dir": 200.0, "solar_direct": 60.0, "solar_diffuse": 20.0}
    base.update(kw)
    return pd.DataFrame({"time": pd.to_datetime([f"{date} 12:30"]), **{k: [v] for k, v in base.items()}})


def test_geometric_solar_needs_direct_or_dni_and_diffuse_columns():
    df = midday("2025-12-21").drop(columns=["solar_direct"])
    with pytest.raises(ValueError, match="solar_model='ghi'"):
        solar_on_conductor(df)
    df["solar_rad"] = 80.0
    assert solar_on_conductor(df, solar_model="ghi")[0] == 80.0


def test_unknown_solar_model_is_rejected():
    with pytest.raises(ValueError):
        solar_on_conductor(midday("2025-12-21"), solar_model="nonsense")


def test_dni_column_is_preferred_over_deriving_it():
    df = midday("2025-12-21")
    df["solar_dni"] = 0.0
    low = solar_on_conductor(df)[0]
    assert low == pytest.approx(20.0 * (1.0 + 0.5 * np.pi * config.GROUND_ALBEDO))   # diffuse only


def test_winter_midday_geometric_model_cuts_ampacity_below_the_horizontal_shortcut():
    line = config.Line(bearing_deg=90.0)                 # east-west line: sun due south is perpendicular
    df = midday("2025-12-21", v_wind=0.5)
    geometric = compute_dlr(df, line=line, solar_model="geometric")[0]
    df["solar_rad"] = df["solar_direct"] + df["solar_diffuse"]
    shortcut = compute_dlr(df, line=line, solar_model="ghi")[0]
    assert geometric < shortcut


def test_line_bearing_changes_the_solar_load():
    df = midday("2025-06-21", solar_direct=500.0, solar_diffuse=100.0)
    east_west = solar_on_conductor(df, config.Line(bearing_deg=90.0))[0]
    north_south = solar_on_conductor(df, config.Line(bearing_deg=0.0))[0]
    assert east_west > north_south


def test_night_hours_carry_no_solar_load():
    df = pd.DataFrame({"time": pd.to_datetime(["2025-12-21 00:30"]), "solar_direct": [0.0], "solar_diffuse": [0.0]})
    assert solar_on_conductor(df)[0] == 0.0


def test_add_dlr_columns_still_reports_headroom_against_the_seasonal_static_rating():
    out = add_dlr_columns(midday("2025-12-21"))
    assert out["season"].iloc[0] == "winter"
    assert out["headroom_ratio"].iloc[0] == pytest.approx(out["dlr_ampacity"].iloc[0] / out["static_rating"].iloc[0])
