"""Hourly DLR versus seasonal static ratings, with a sheltering-factor sensitivity table."""

import pandas as pd

from dlr import config
from dlr.pipeline import add_dlr_columns, headroom_summary, load_weather

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")


def run_headroom_analysis():
    weather = load_weather()
    df = add_dlr_columns(weather)

    print("Static ratings (A): " + ", ".join(f"{k} {v:.0f}" for k, v in df.attrs["static_ratings"].items()))
    print(f"\nHeadroom over seasonal static rating (sheltering factor {config.SHELTERING_FACTOR}):")
    print(headroom_summary(df))

    print("\nSensitivity to the effective-wind (sheltering) factor, whole year:")
    rows = {}
    for f in config.SHELTERING_SWEEP:
        r = add_dlr_columns(weather, sheltering=f)["headroom_ratio"]
        rows[f"x{f}"] = {"median_%": (r.median() - 1) * 100,
                         "P5_%": (r.quantile(0.05) - 1) * 100,
                         "P1_%": (r.quantile(0.01) - 1) * 100,
                         "below_static_%": (r < 1.0).mean() * 100}
    print(pd.DataFrame(rows).T)


if __name__ == "__main__":
    run_headroom_analysis()
    
