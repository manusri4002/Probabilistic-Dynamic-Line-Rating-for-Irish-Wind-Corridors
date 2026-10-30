"""Hourly DLR versus seasonal static ratings, reported as bands over the unknown assumptions.

Line bearing, sheltering factor and sky-diffuse model are assumptions, not measurements, so the
headline is the WORST span and a range, not one number (see dlr/bands.py)."""

import pandas as pd

from dlr import config
from dlr.bands import band, bearing_profile, scenario_table, worst_span
from dlr.pipeline import add_dlr_columns, headroom_summary, load_weather

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")


def run_headroom_analysis():
    weather = load_weather()
    df = add_dlr_columns(weather)

    print("Static ratings (A): " + ", ".join(f"{k} {v:.0f}" for k, v in df.attrs["static_ratings"].items()))
    print(f"\nCENTRAL CASE ONLY (bearing {config.LINE.bearing_deg:.0f} deg, sheltering {config.SHELTERING_FACTOR}, "
          f"{config.DIFFUSE_MODEL}); an assumption, not a result:")
    print(headroom_summary(df))

    table = scenario_table(weather)
    print(f"\nBy line bearing (sheltering {config.SHELTERING_FACTOR}, {config.DIFFUSE_MODEL}). "
          "P5 is often tied across bearings because calm hours set it; the median shows the wind-angle effect:")
    print(bearing_profile(table).drop(columns=["below_static_pct"]))

    worst = worst_span(table)
    print("\nWorst span (lowest P5 bearing) for each sheltering factor and diffuse model:")
    print(worst.rename(columns={"bearing_deg": "worst_bearing"}).drop(columns=["below_static_pct"]).to_string(index=False))

    shelters = sorted(config.SHELTERING_SWEEP)
    print(f"\nBAND for the worst span over sheltering {shelters[0]}-{shelters[-1]} and "
          f"{' / '.join(config.DIFFUSE_MODELS)} diffuse models. Quote these ranges, not one sheltering factor:")
    print(band(worst))


if __name__ == "__main__":
    run_headroom_analysis()
