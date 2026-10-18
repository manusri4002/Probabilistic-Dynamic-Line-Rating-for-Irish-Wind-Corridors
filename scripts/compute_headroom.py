"""Compute hourly DLR ampacity versus the static baseline rating."""

from dlr.pipeline import add_dlr_columns, load_weather


def run_headroom_analysis():
    df = add_dlr_columns(load_weather())
    static = df.attrs["static_rating"]

    print(f"Baseline static rating: {static:.1f} A")
    print(f"Mean DLR ampacity: {df['dlr_ampacity'].mean():.1f} A")
    print(f"Median headroom over static: {df['headroom_pct'].median():.1f}%")
    print(f"Time above static rating: {(df['dlr_ampacity'] > static).mean() * 100:.1f}%")
    print(f"5th-percentile headroom: {df['headroom_pct'].quantile(0.05):.1f}%")


if __name__ == "__main__":
    run_headroom_analysis()
    