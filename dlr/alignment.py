"""Checks that weather and wind-output timestamps line up.

Two independent checks, because a lag in the correlation alone cannot tell a time-zone
mistake from a real physical lead:
  1. sunshine peak: in UTC data the daily sunshine peak falls on the same hour in winter
     and in summer. If summer rows were in Irish local time it would read one hour later.
  2. lag scan: how the correlation between weather and wind output changes with a time shift.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

LAGS_H = range(-12, 13)
MIN_HOURS = 100


def spearman(a, b):
    """Rank correlation. p-values are deliberately not reported: hourly weather is strongly
    autocorrelated, so the effective sample size is far below the row count."""
    return stats.spearmanr(a, b).statistic


def join_weather_wind(weather, wind, lag_h=0):
    """Inner-join on UTC hour. lag_h=k pairs each weather hour with the wind output from
    k hours EARLIER (k=0 is the same hour)."""
    shifted = wind.assign(time=wind["time"] + pd.Timedelta(hours=lag_h))
    return weather.merge(shifted, on="time", how="inner")


def lag_scan(weather, wind, lags=LAGS_H):
    """Spearman(weather wind speed, ROI wind output) for each lag, as a Series indexed by lag."""
    out = {}
    for k in lags:
        m = join_weather_wind(weather, wind, k)
        out[k] = spearman(m["v_wind"], m["ie_wind_mw"]) if len(m) >= MIN_HOURS else float("nan")
    return pd.Series(out, name="spearman")


def is_irish_summer_time(times):
    """Boolean array: True where a naive-UTC timestamp falls in Irish summer time (UTC+1)."""
    utc = pd.DatetimeIndex(times)
    local = utc.tz_localize("UTC").tz_convert("Europe/Dublin").tz_localize(None)
    return np.asarray(local - utc > pd.Timedelta(0))


def solar_peak_hours(weather, column="solar_rad"):
    """{'winter': hour, 'summer': hour}: the UTC hour with the highest mean sunshine, for each
    season that has at least two weeks of data. Equal values mean the timestamps are UTC."""
    summer = is_irish_summer_time(weather["time"])
    out = {}
    for name, mask in (("winter", ~summer), ("summer", summer)):
        sub = weather[mask]
        if len(sub) >= 24 * 14:
            out[name] = int(sub.groupby(sub["time"].dt.hour)[column].mean().idxmax())
    return out