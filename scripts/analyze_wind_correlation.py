"""Does DLR headroom coincide with high wind output? Uses real EirGrid ROI wind generation.

One ERA5 grid point (north Dublin) is compared with national output, so this measures how
well a single corridor's weather tracks the fleet, not the output of any particular wind farm.
"""

import pandas as pd
import scipy.stats as stats

from dlr.eirgrid import load_wind_hourly
from dlr.pipeline import add_dlr_columns, load_weather

LAGS_H = range(-3, 4)


def spearman(a, b):
    """Rank correlation. p-values are deliberately not reported: hourly weather is strongly
    autocorrelated, so the effective sample size is far below the row count."""
    return stats.spearmanr(a, b).statistic


def join_weather_wind(weather, wind, lag_h=0):
    """Inner-join on UTC hour. lag_h=k pairs each weather hour with the wind output from
    k hours EARLIER (k=0 is the correct alignment)."""
    shifted = wind.assign(time=wind["time"] + pd.Timedelta(hours=lag_h))
    return weather.merge(shifted, on="time", how="inner")


def run_correlation_analysis():
    weather = add_dlr_columns(load_weather())
    wind = load_wind_hourly()
    m = join_weather_wind(weather, wind)

    print(f"Weather hours: {len(weather)} | wind hours: {len(wind)} | matched: {len(m)}")
    if len(m) < 0.9 * min(len(weather), len(wind)):
        raise SystemExit("Fewer than 90% of hours matched: the two datasets cover different periods.")

    # Sanity check 1: does reanalysis wind at one point track national output at all?
    print(f"\nSanity: Spearman(ERA5 10 m wind, ROI wind output) = {spearman(m['v_wind'], m['ie_wind_mw']):.3f}")
    # Sanity check 2: alignment. The correlation should peak at lag 0 if timestamps line up.
    scan = {}
    for k in LAGS_H:
        mk = join_weather_wind(weather, wind, lag_h=k)
        scan[k] = spearman(mk["v_wind"], mk["ie_wind_mw"])
    print("Alignment scan, Spearman(ERA5 wind, ROI output) by lag in hours (peak should be 0):")
    for k, r in scan.items():
        print(f"  lag {k:+d} h: {r:.3f}")
    best = max(scan, key=scan.get)
    if best != 0:
        print(f"WARNING: correlation peaks at lag {best:+d} h, not 0. The weather and EirGrid "
              "timestamps are probably offset: check time zones before trusting anything below.")

    rho = spearman(m["ie_wind_mw"], m["headroom_ratio"])
    print(f"\nSpearman(ROI wind output, DLR/static headroom ratio) = {rho:.3f}")

    m["wind_decile"] = pd.qcut(m["ie_wind_mw"], 10, labels=False) + 1
    table = m.groupby("wind_decile").agg(
        wind_MW=("ie_wind_mw", "median"),
        headroom_ratio=("headroom_ratio", "median"),
        below_static_pct=("headroom_ratio", lambda r: (r < 1.0).mean() * 100),
        hours=("headroom_ratio", "size"),
    )
    print("\nBy ROI wind-output decile (1 = lowest output):")
    print(table.round(2).to_string())

    high = m[m["ie_wind_mw"] >= m["ie_wind_mw"].quantile(0.80)]
    derated = m[m["headroom_ratio"] < 1.0]
    print(f"\nHigh-output hours (top 20%): {len(high)}; of these, DLR below static in "
          f"{(high['headroom_ratio'] < 1.0).sum()} ({(high['headroom_ratio'] < 1.0).mean() * 100:.2f}%)")
    print(f"All hours: DLR below static in {len(derated)} ({len(derated) / len(m) * 100:.2f}%)")
    if len(derated):
        share = (derated["ie_wind_mw"] >= m["ie_wind_mw"].quantile(0.60)).mean() * 100
        print(f"Of the {len(derated)} derated hours, {share:.0f}% occurred when ROI wind output was in the top 40%")


if __name__ == "__main__":
    run_correlation_analysis()
