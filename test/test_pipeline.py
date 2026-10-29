import numpy as np
import pandas as pd
import pytest

from dlr import config
from dlr.pipeline import (
    add_dlr_columns,
    attack_angle_deg,
    load_weather,
    static_rating,
    wind_at_conductor,
)


@pytest.mark.parametrize("wind_dir,bearing,expected", [
    (45, 45, 0),      # along the line
    (225, 45, 0),     # same axis, opposite direction
    (135, 45, 90),    # perpendicular
    (315, 45, 90),
    (0, 45, 45),
    (90, 45, 45),
    (350, 45, 55),    # wraps through north
])
def test_attack_angle_folding(wind_dir, bearing, expected):
    assert attack_angle_deg(wind_dir, bearing) == pytest.approx(expected)


def test_shear_scales_ten_metre_wind_to_conductor_height():
    factor = (config.LINE.conductor_height_m / config.MET_HEIGHT_M) ** config.SHEAR_ALPHA
    assert wind_at_conductor([5.0])[0] == pytest.approx(5.0 * factor)
    assert wind_at_conductor([5.0], apply_shear=False)[0] == pytest.approx(5.0)


def test_calm_wind_is_floored():
    assert wind_at_conductor([0.0])[0] == config.WIND_FLOOR_MS


def test_static_rating_regression():
    assert static_rating() == pytest.approx(864.1, abs=0.5)


def test_add_dlr_columns_shapes_and_headroom_definition():
    df = pd.DataFrame({"time": pd.date_range("2023-07-01", periods=2, freq="h"),
                       "t_ambient": [10.0, 20.0], "v_wind": [5.0, 1.0],
                       "wind_dir": [135.0, 45.0],
                       "solar_direct": [0.0, 400.0], "solar_diffuse": [0.0, 100.0]})
    out = add_dlr_columns(df)
    assert {"dlr_ampacity", "headroom_ratio", "headroom_pct"} <= set(out.columns)
    assert out["headroom_pct"].iloc[0] == pytest.approx((out["headroom_ratio"].iloc[0] - 1) * 100)
    assert out["dlr_ampacity"].iloc[0] > out["dlr_ampacity"].iloc[1]


def test_load_weather_rejects_kmh_data(tmp_path):
    """Regression for the original bug: km/h values read as m/s."""
    n = 48
    bad = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=n, freq="h"),
                        "t_ambient": 8.0, "v_wind": np.full(n, 16.0),
                        "wind_dir": 200.0, "solar_rad": 0.0})
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="implausible"):
        load_weather(path)
