import numpy as np
import pandas as pd

from dlr.forecast import (FEATURES, TARGET, blocked_split, build_dataset, calibrate, climatology_quantiles,
                          conformal_shift, day_block_bootstrap, pinball_loss, predict_quantiles)


def make_weather(days=40, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    n = len(time)
    df = pd.DataFrame({
        "time": time,
        "t_ambient": 8 + rng.normal(0, 2, n),
        "v_wind": np.abs(rng.normal(5, 2, n)),
        "wind_dir": rng.uniform(0, 360, n),
        TARGET: rng.normal(1500, 300, n),
    })
    df["season"] = "winter"
    df["static_rating"] = 1043.0
    return df


def test_every_target_is_issued_at_noon_the_day_before():
    ds = build_dataset(make_weather())
    assert ((ds["time"].dt.normalize() - ds["issue_time"].dt.normalize()) == pd.Timedelta(days=1)).all()
    assert (ds["issue_time"].dt.hour == 12).all()
    assert ds["lead_h"].min() == 12 and ds["lead_h"].max() == 35


def test_features_do_not_change_when_the_future_is_scrambled():
    """The leakage test: for every hour of one day (early and late leads behave differently),
    rewrite every observation after that target's issue time and check its features are untouched."""
    df = make_weather()
    ds = build_dataset(df)
    day = ds[ds["time"].dt.normalize() == pd.Timestamp("2026-01-20")]
    assert len(day) == 24
    for _, row in day.iterrows():
        scrambled = df.copy()
        future = scrambled["time"] > row["issue_time"]
        rng = np.random.default_rng(99)
        for col in ["t_ambient", "v_wind", "wind_dir", TARGET]:
            scrambled.loc[future, col] = rng.normal(1000, 1, future.sum())
        row2 = build_dataset(scrambled).query("time == @row.time").iloc[0]
        assert np.allclose(row[FEATURES].to_numpy(float), row2[FEATURES].to_numpy(float)), row["time"]


def test_features_do_change_when_the_past_is_scrambled():
    """Guard for the test above: it would pass trivially if the features ignored the data."""
    df = make_weather()
    ds = build_dataset(df)
    row = ds.iloc[300]
    scrambled = df.copy()
    scrambled.loc[scrambled["time"] <= row["issue_time"], "v_wind"] += 3.0
    row2 = build_dataset(scrambled).query("time == @row.time").iloc[0]
    assert row2["v_mean24"] != row["v_mean24"]


def test_rows_with_missing_hours_are_dropped_not_filled():
    df = make_weather()
    gappy = df.drop(df.index[200:210])
    ds = build_dataset(gappy)
    assert not ds[FEATURES].isna().any().any()
    assert len(ds) < len(build_dataset(df))


def test_pinball_loss_known_values():
    assert pinball_loss([1.0], [0.0], 0.9) == 0.9      # under-forecast costs q per unit
    assert np.isclose(pinball_loss([1.0], [2.0], 0.9), 0.1)   # over-forecast costs 1-q per unit


def test_split_sets_are_disjoint_and_cover_all_seasons():
    times = pd.date_range("2026-01-01", "2026-09-30 23:00", freq="h")
    train, cal, test = blocked_split(times)
    assert not (train & cal).any() and not (train & test).any() and not (cal & test).any()
    for mask in (train, cal, test):
        assert pd.DatetimeIndex(times[mask]).month.nunique() >= 5
    assert train.sum() > cal.sum() and train.sum() > test.sum()


def test_split_keeps_a_gap_between_neighbouring_blocks():
    times = pd.date_range("2026-01-01", "2026-09-30 23:00", freq="h")
    masks = blocked_split(times)
    day = pd.Series(pd.DatetimeIndex(times).normalize())
    sets = np.full(len(times), -1)
    for s, m in enumerate(masks):
        sets[m] = s
    per_day = pd.DataFrame({"day": day, "set": sets}).groupby("day")["set"].agg(lambda x: x.iloc[0])
    arr = per_day.to_numpy()
    for i in range(1, len(arr)):
        if arr[i] >= 0 and arr[i - 1] >= 0:
            assert arr[i] == arr[i - 1]      # two adjacent kept days from different sets never touch


def test_conformal_lower_quantile_reaches_nominal_coverage():
    rng = np.random.default_rng(3)
    y_cal, y_new = rng.normal(0, 1, 2000), rng.normal(0, 1, 20000)
    bad_pred = 0.5                                   # a deliberately over-optimistic P5 (true P5 is -1.64)
    shift = conformal_shift(0.05, np.full(2000, bad_pred), y_cal)
    fixed = bad_pred + shift
    coverage = np.mean(y_new >= fixed)               # share of new values at or above the forecast
    assert shift < 0 and coverage >= 0.95 - 0.01
    assert abs(coverage - 0.95) < 0.02               # and not wildly conservative


def test_conformal_upper_quantile_and_median():
    rng = np.random.default_rng(4)
    y_cal, y_new = rng.normal(0, 1, 2000), rng.normal(0, 1, 20000)
    shift = conformal_shift(0.9, np.full(2000, 0.0), y_cal)
    assert abs(np.mean(y_new <= shift) - 0.9) < 0.02
    assert conformal_shift(0.5, np.zeros(10), np.ones(10)) == 0.0


def test_conformal_refuses_a_calibration_set_that_is_too_small():
    import pytest
    with pytest.raises(ValueError):
        conformal_shift(0.01, np.zeros(20), np.zeros(20))


def test_calibrate_keeps_quantiles_ordered():
    cal = pd.DataFrame({0.1: [0.0] * 50, 0.9: [1.0] * 50})
    y = np.linspace(-3, 4, 50)
    test = pd.DataFrame({0.1: [0.0, 0.1], 0.9: [1.0, 1.1]})
    out = calibrate(cal, y, test)
    assert (out[0.1] <= out[0.9]).all()


def test_predict_quantiles_never_crosses():
    class Fake:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    out = predict_quantiles({0.1: Fake(5.0), 0.9: Fake(2.0)}, np.zeros((3, 1)))
    assert (out[0.1] <= out[0.9]).all()


def test_climatology_uses_training_rows_only():
    train = pd.DataFrame({"hour": [0] * 4, "season": ["winter"] * 4, TARGET: [10.0, 20.0, 30.0, 40.0]})
    target = pd.DataFrame({"hour": [0, 0], "season": ["winter", "summer"], TARGET: [999.0, 999.0]})
    out = climatology_quantiles(train, target, [0.5])
    assert out[0.5].tolist() == [25.0, 25.0]        # summer is unseen: falls back to the hour-of-day value


def test_day_block_bootstrap_interval_is_wider_than_a_binomial_one_for_clustered_flags():
    n_days = 60
    day_flag = np.random.default_rng(5).random(n_days) < 0.1
    flags = np.repeat(day_flag, 24)                  # a bad day is bad for all 24 hours
    times = pd.date_range("2026-01-01", periods=n_days * 24, freq="h")
    lo, hi = day_block_bootstrap(flags, times)
    p = flags.mean()
    binom_half = 1.96 * np.sqrt(p * (1 - p) / len(flags))
    assert (hi - lo) / 2 > 3 * binom_half