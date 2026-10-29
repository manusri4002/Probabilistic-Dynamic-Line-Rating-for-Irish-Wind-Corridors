import numpy as np
import pandas as pd

from dlr import config
from dlr.pipeline import add_dlr_columns, attack_angle_deg, load_weather


def runs_of_true(flags):
    """Lengths of consecutive True runs (hours) in a boolean array."""
    flags = np.asarray(flags, bool)
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False].astype(int)))
    return edges[1::2] - edges[::2]


def run():
    df = add_dlr_columns(load_weather())
    n = len(df)
    print(f"Period: {df['time'].min()} to {df['time'].max()} ({n} hours, {n / 24:.0f} days)")

    months = df["time"].dt.to_period("M").value_counts().sort_index()
    print("\nHours per month (a full month is 672-744):")
    print(months.to_string())
    absent = sorted(set(range(1, 13)) - set(df["time"].dt.month))
    print(f"Calendar months with NO data: {absent if absent else 'none'}")

    print("\nAmbient temperature (degC): "
          f"min {df['t_ambient'].min():.1f}, max {df['t_ambient'].max():.1f}; "
          f"hours below 0: {(df['t_ambient'] < 0).sum()}, above 20: {(df['t_ambient'] > 20).sum()}")
    print(f"Wind at 10 m (m/s): hours below 1: {(df['v_wind'] < 1).sum()} "
          f"({(df['v_wind'] < 1).mean() * 100:.1f}%), below 2: {(df['v_wind'] < 2).sum()}, "
          f"max {df['v_wind'].max():.1f}")

    derate = (df["headroom_ratio"] < 1.0).to_numpy()
    runs = runs_of_true(derate)
    print(f"\nHours with DLR below static: {derate.sum()} in {len(runs)} separate events; "
          f"longest event {runs.max() if len(runs) else 0} h, median {np.median(runs) if len(runs) else 0:.0f} h")
    print("The P1/P5 headroom numbers are decided by these events, not by the thousands of ordinary hours.")
    if len(runs):
        ev_months = df.loc[derate, "time"].dt.to_period("M").value_counts().sort_index()
        print("Derated hours by month:\n" + ev_months.to_string())

    angle = attack_angle_deg(df["wind_dir"].to_numpy(), config.LINE.bearing_deg)
    bins = pd.cut(angle, [0, 30, 60, 90], include_lowest=True, labels=["0-30 (along line)", "30-60", "60-90 (across)"])
    print("\nWind angle to the line (only one assumed bearing):")
    print(pd.Series(bins).value_counts(normalize=True).sort_index().mul(100).round(1).to_string())

    warm_calm = ((df["t_ambient"] > 15) & (df["v_wind"] < 2)).sum()
    print(f"\nWarm and calm hours (above 15 degC and below 2 m/s): {warm_calm} "
          f"({warm_calm / n * 100:.1f}% of the record). Little wind and high ambient is where a line is thermally tightest.")


if __name__ == "__main__":
    run()