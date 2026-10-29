"""The forecast target is ampacity in amps, so it cannot step at season boundaries."""

import pandas as pd

from dlr.forecast import FEATURES, TARGET, build_dataset
from dlr.pipeline import add_dlr_columns


def constant_weather():
    """Identical weather for 20 days across 1 March, with no sun so nothing varies by date."""
    time = pd.date_range("2026-02-20", "2026-03-11 23:00", freq="h")
    return pd.DataFrame({"time": time, "t_ambient": 8.0, "v_wind": 5.0, "wind_dir": 200.0,
                         "solar_direct": 0.0, "solar_diffuse": 0.0})


def test_target_is_ampacity_and_no_feature_is_a_headroom_percentage():
    assert TARGET == "dlr_ampacity"
    assert not any("headroom" in f for f in FEATURES)


def test_old_headroom_target_steps_at_the_season_boundary_but_the_ampacity_target_does_not():
    df = add_dlr_columns(constant_weather())
    assert df["headroom_pct"].nunique() > 1                       # the defect: 28 Feb vs 1 Mar
    jump = (df.loc[df["time"] >= "2026-03-01", "headroom_pct"].iloc[0]
            - df.loc[df["time"] < "2026-03-01", "headroom_pct"].iloc[-1])
    assert abs(jump) > 5.0

    ds = build_dataset(df)
    assert ds["time"].min() < pd.Timestamp("2026-03-01") < ds["time"].max()
    assert ds[TARGET].nunique() == 1                              # same weather, same target
    assert {"static_rating", "season"} <= set(ds.columns)         # headroom can still be derived
