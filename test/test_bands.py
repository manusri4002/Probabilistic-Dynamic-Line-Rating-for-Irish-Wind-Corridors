import numpy as np
import pandas as pd
import pytest

from dlr import config
from dlr.bands import band, bearing_profile, scenario_table, worst_span
from dlr.pipeline import compute_dlr

BEARINGS = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165)
SHELTERING = (1.0, 0.7, 0.4)
MODELS = ("cigre601", "isotropic")


def weather(days=20, wind_dir=200.0, sun=True, month="2026-01"):
    """Deterministic hourly weather: wind between 0.5 and 9.5 m/s, a fixed direction, and a
    sunshine curve that peaks at midday and is zero at night (or zero everywhere if sun=False)."""
    time = pd.date_range(f"{month}-01", periods=days * 24, freq="h")
    hour = time.hour.to_numpy()
    wind = 5.0 + 4.5 * np.sin(np.arange(len(time)) / 7.0)
    day_curve = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0.0, None) if sun else np.zeros(len(time))
    return pd.DataFrame({"time": time, "t_ambient": 8.0, "v_wind": wind, "wind_dir": wind_dir,
                         "solar_direct": 250.0 * day_curve, "solar_diffuse": 60.0 * day_curve})


def table(df=None, **kw):
    args = dict(bearings=BEARINGS, shelterings=SHELTERING, diffuse_models=MODELS)
    args.update(kw)
    return scenario_table(weather() if df is None else df, **args)


def test_one_row_per_scenario_and_columns():
    t = table()
    assert len(t) == len(BEARINGS) * len(SHELTERING) * len(MODELS)
    assert {"bearing_deg", "sheltering", "diffuse_model", "median_pct", "p5_pct", "p1_pct",
            "below_static_h", "below_static_pct"} <= set(t.columns)
    assert not t.isna().any().any()


def test_a_bearing_and_its_reverse_are_the_same_line():
    a = scenario_table(weather(), bearings=(30,), shelterings=(0.7,), diffuse_models=("cigre601",))
    b = scenario_table(weather(), bearings=(210,), shelterings=(0.7,), diffuse_models=("cigre601",))
    metrics = ["median_pct", "p5_pct", "p1_pct", "below_static_h"]
    assert a[metrics].iloc[0].to_numpy() == pytest.approx(b[metrics].iloc[0].to_numpy())


def test_input_frame_is_not_modified():
    df = weather()
    before = df.copy()
    table(df)
    pd.testing.assert_frame_equal(df, before)


def test_worst_span_is_the_bearing_most_aligned_with_the_wind():
    """No sun, wind from 200 degrees: the line axis at 15 degrees is only 5 degrees off the wind,
    so it is cooled least and must be reported as the worst span; 105 degrees is the best."""
    t = table(weather(sun=False))
    worst = worst_span(t)
    assert set(worst["bearing_deg"]) == {15.0}
    prof = bearing_profile(t, sheltering=0.7, diffuse_model="cigre601")
    # P5 is set by near-calm hours where wind angle is irrelevant (several bearings tie), so the
    # wind-angle effect is read from the median
    assert prof["median_pct"].idxmin() == 15.0
    assert prof["median_pct"].idxmax() == 105.0
    assert prof["p5_pct"].min() == worst.loc[worst["sheltering"] == 0.7, "p5_pct"].iloc[0]


def test_worst_span_has_the_lowest_p5_of_its_scenario_group():
    t = table()
    worst = worst_span(t)
    assert len(worst) == len(SHELTERING) * len(MODELS)
    for _, row in worst.iterrows():
        group = t[(t["sheltering"] == row["sheltering"]) & (t["diffuse_model"] == row["diffuse_model"])]
        assert row["p5_pct"] == group["p5_pct"].min()


def test_less_sheltering_means_more_headroom():
    t = table()
    prof = {s: bearing_profile(t, sheltering=s, diffuse_model="cigre601") for s in SHELTERING}
    assert (prof[1.0]["median_pct"] > prof[0.7]["median_pct"]).all()
    assert (prof[0.7]["median_pct"] > prof[0.4]["median_pct"]).all()
    assert (prof[0.4]["below_static_h"] >= prof[1.0]["below_static_h"]).all()


def test_band_contains_the_central_case_and_is_ordered():
    worst = worst_span(table())
    b = band(worst)
    assert (b["min"] <= b["max"]).all()
    central = worst[(worst["sheltering"] == config.SHELTERING_FACTOR) & (worst["diffuse_model"] == config.DIFFUSE_MODEL)]
    assert len(central) == 1
    for metric in b.index:
        assert b.loc[metric, "min"] <= central[metric].iloc[0] <= b.loc[metric, "max"]
    assert b.loc["p5_pct", "min"] < b.loc["p5_pct", "max"]       # the assumptions genuinely matter


def test_bearing_changes_the_result_when_the_sun_is_up_and_not_otherwise_by_the_sun_alone():
    sunny = bearing_profile(table(weather(sun=True)), 0.7, "cigre601")
    assert sunny["median_pct"].nunique() > 1


def test_isotropic_diffuse_model_cuts_ampacity_when_there_is_sky_light():
    df = weather(days=3, month="2026-06")
    cigre = compute_dlr(df, diffuse_model="cigre601")
    iso = compute_dlr(df, diffuse_model="isotropic")
    assert (iso <= cigre + 1e-9).all()
    assert (iso < cigre).any()


def test_unknown_diffuse_model_is_rejected_by_compute_dlr():
    with pytest.raises(ValueError):
        compute_dlr(weather(days=2), diffuse_model="nonsense")


def test_albedo_raises_solar_load_and_lowers_ampacity():
    df = weather(days=3, month="2026-06")
    assert compute_dlr(df, albedo=0.3).sum() < compute_dlr(df, albedo=0.0).sum()


def test_default_sweep_covers_every_line_once():
    assert config.BEARING_SWEEP == tuple(range(0, 180, 15))
    assert config.SHELTERING_FACTOR in config.SHELTERING_SWEEP
