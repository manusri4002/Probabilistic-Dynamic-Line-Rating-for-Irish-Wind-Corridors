import pandas as pd
import pytest

from dlr import config
from dlr.pipeline import (
    add_dlr_columns,
    compute_dlr,
    headroom_summary,
    seasonal_static_ratings,
    wind_at_conductor,
)


def hourly(month, **kw):
    base = {"t_ambient": 10.0, "v_wind": 5.0, "wind_dir": 135.0, "solar_rad": 200.0}
    base.update(kw)
    return pd.DataFrame({"time": pd.date_range(f"2023-{month:02d}-10", periods=3, freq="h"),
                         **{k: [v] * 3 for k, v in base.items()}})


def test_every_month_has_a_season():
    assert sorted(config.SEASON_BY_MONTH) == list(range(1, 13))
    assert set(config.SEASON_BY_MONTH.values()) == set(config.SEASONS)


def test_colder_season_has_higher_static_rating():
    r = seasonal_static_ratings()
    assert r["winter"] > r["spring_autumn"] > r["summer"]


def test_headroom_is_measured_against_the_rows_own_season():
    ratings = seasonal_static_ratings()
    for month, season in [(1, "winter"), (4, "spring_autumn"), (7, "summer")]:
        out = add_dlr_columns(hourly(month))
        assert (out["season"] == season).all()
        assert out["static_rating"].iloc[0] == pytest.approx(ratings[season])
        assert out["headroom_ratio"].iloc[0] == pytest.approx(out["dlr_ampacity"].iloc[0] / ratings[season])


def test_sheltering_lowers_wind_and_rating():
    assert wind_at_conductor([6.0], sheltering=0.5)[0] == pytest.approx(
        0.5 * wind_at_conductor([6.0], sheltering=1.0)[0])
    df = hourly(1)
    assert compute_dlr(df, sheltering=0.5).iloc[0] < compute_dlr(df, sheltering=1.0).iloc[0]


def test_missing_time_column_is_rejected():
    with pytest.raises(ValueError, match="time"):
        add_dlr_columns(hourly(1).drop(columns="time"))


def test_summary_flags_hours_below_static():
    calm = add_dlr_columns(hourly(7, v_wind=0.0, solar_rad=1000.0, wind_dir=45.0))
    windy = add_dlr_columns(hourly(7, v_wind=12.0))
    assert headroom_summary(calm).loc["all year", "below_static_%"] == 100.0
    assert headroom_summary(windy).loc["all year", "below_static_%"] == 0.0
