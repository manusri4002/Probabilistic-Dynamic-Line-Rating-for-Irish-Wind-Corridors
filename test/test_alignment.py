import numpy as np
import pandas as pd

from dlr.alignment import is_irish_summer_time, join_weather_wind, lag_scan, solar_peak_hours


def test_summer_time_boundaries_2026():
    t = pd.to_datetime(["2026-03-29 00:59", "2026-03-29 01:00", "2026-10-25 00:59", "2026-10-25 01:00"])
    assert is_irish_summer_time(t).tolist() == [False, True, True, False]


def test_join_pairs_weather_with_wind_from_k_hours_earlier():
    wind = pd.DataFrame({"time": [pd.Timestamp("2026-01-01 10:00")], "ie_wind_mw": [5.0]})
    weather = pd.DataFrame({"time": [pd.Timestamp("2026-01-01 12:00")], "v_wind": [1.0]})
    assert join_weather_wind(weather, wind, lag_h=2)["ie_wind_mw"].tolist() == [5.0]
    assert join_weather_wind(weather, wind, lag_h=0).empty


def test_lag_scan_recovers_a_known_lead():
    rng = np.random.default_rng(1)
    times = pd.date_range("2026-01-01", periods=2000, freq="h")
    out = pd.Series(rng.normal(size=2000)).ewm(span=12).mean().to_numpy()    # smooth, like real output
    wind = pd.DataFrame({"time": times, "ie_wind_mw": out})
    # weather follows the national output by 3 hours (weather at t looks like output at t-3)
    weather = pd.DataFrame({"time": times, "v_wind": np.roll(out, 3) + rng.normal(0, 0.01, 2000)})
    scan = lag_scan(weather, wind)
    assert scan.idxmax() == 3


def synthetic_sunshine(local_label_in_summer=False):
    utc = pd.date_range("2026-01-01", "2026-09-30 23:00", freq="h")
    hour_mid = utc.hour.to_numpy() - 0.5            # preceding-hour mean is centred half an hour earlier
    solar = np.clip(np.cos(np.pi * (hour_mid - 12.42) / 16.0), 0, None) * 500   # noon at 12:25 UTC
    labels = pd.Series(utc)
    if local_label_in_summer:
        labels = labels + pd.to_timedelta(is_irish_summer_time(utc).astype(int), unit="h")
    return pd.DataFrame({"time": labels.to_numpy(), "solar_rad": solar})


def test_sunshine_peak_is_the_same_in_both_seasons_for_utc_data():
    peaks = solar_peak_hours(synthetic_sunshine())
    assert peaks["winter"] == peaks["summer"]


def test_sunshine_peak_shows_a_local_time_label():
    peaks = solar_peak_hours(synthetic_sunshine(local_label_in_summer=True))
    assert peaks["summer"] == peaks["winter"] + 1