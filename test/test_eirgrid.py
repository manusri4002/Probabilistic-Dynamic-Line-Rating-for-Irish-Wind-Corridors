import pandas as pd
import pytest
from dlr.eirgrid import to_hourly_utc


def quarter_hours(start, n, offset, mw, avail=None):
    return pd.DataFrame({
        "DateTime": pd.date_range(start, periods=n, freq="15min"),
        "GMT Offset": offset,
        "IE Wind Generation": mw,
        "IE Wind Availability": mw if avail is None else avail,
    })


def test_summer_local_time_is_converted_to_utc():
    out = to_hourly_utc(quarter_hours("2026-07-01 01:00", 4, offset=1, mw=100.0))
    assert out["time"].tolist() == [pd.Timestamp("2026-07-01 00:00")]


def test_hourly_value_is_the_mean_of_four_quarter_hours():
    raw = quarter_hours("2026-01-10 00:00", 4, offset=0, mw=[100.0, 200.0, 300.0, 400.0])
    assert to_hourly_utc(raw)["ie_wind_mw"].iloc[0] == pytest.approx(250.0)


def test_incomplete_hours_are_dropped():
    out = to_hourly_utc(quarter_hours("2026-01-10 00:00", 5, offset=0, mw=100.0))
    assert len(out) == 1


def test_missing_reading_drops_that_hour():
    raw = quarter_hours("2026-01-10 00:00", 8, offset=0, mw=[1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = to_hourly_utc(raw)
    assert out["time"].tolist() == [pd.Timestamp("2026-01-10 01:00")]


def test_spring_forward_gap_leaves_a_continuous_utc_series():
    """29 Mar 2026: local 01:00-01:59 does not exist. 00:xx is GMT, 02:xx is IST."""
    raw = pd.concat([quarter_hours("2026-03-29 00:00", 4, offset=0, mw=10.0),
                     quarter_hours("2026-03-29 02:00", 4, offset=1, mw=20.0)], ignore_index=True)
    out = to_hourly_utc(raw)
    assert out["time"].tolist() == [pd.Timestamp("2026-03-29 00:00"), pd.Timestamp("2026-03-29 01:00")]
    assert out["ie_wind_mw"].tolist() == [10.0, 20.0]


def autumn_raw(with_offset=True):
    """25 Oct 2026: local 01:00-01:59 happens twice (IST then GMT)."""
    first = quarter_hours("2026-10-25 01:00", 4, offset=1, mw=10.0)
    second = quarter_hours("2026-10-25 01:00", 4, offset=0, mw=20.0)
    raw = pd.concat([quarter_hours("2026-10-25 00:00", 4, offset=1, mw=5.0), first, second,
                     quarter_hours("2026-10-25 02:00", 4, offset=0, mw=30.0)], ignore_index=True)
    return raw if with_offset else raw.drop(columns="GMT Offset")


def test_autumn_repeated_hour_gives_two_distinct_utc_hours():
    out = to_hourly_utc(autumn_raw())
    assert out["time"].tolist() == [pd.Timestamp("2026-10-24 23:00"), pd.Timestamp("2026-10-25 00:00"),
                                    pd.Timestamp("2026-10-25 01:00"), pd.Timestamp("2026-10-25 02:00")]
    assert out["ie_wind_mw"].tolist() == [5.0, 10.0, 20.0, 30.0]


def test_dst_rules_fallback_matches_the_offset_column():
    pd.testing.assert_frame_equal(to_hourly_utc(autumn_raw(with_offset=True)),
                                  to_hourly_utc(autumn_raw(with_offset=False)))


def test_duplicate_utc_timestamps_are_rejected():
    raw = pd.concat([quarter_hours("2026-01-10 00:00", 4, 0, 1.0)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        to_hourly_utc(raw)


def test_missing_wind_column_is_reported():
    with pytest.raises(ValueError, match="IE Wind Generation"):
        to_hourly_utc(quarter_hours("2026-01-10", 4, 0, 1.0).drop(columns="IE Wind Generation"))