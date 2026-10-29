"""EirGrid 'System Data Qtr Hourly' workbooks -> hourly Republic-of-Ireland wind series in UTC.
Source format (checked against the 2026 workbook): sheet 'System Data', 15-minute rows.
DateTime is Irish LOCAL clock time marking the START of each 15-minute period; GMT Offset is
0 in winter and 1 in summer. The hour skipped in spring is simply missing, and the hour
repeated in autumn appears twice, so UTC = DateTime - GMT Offset is unambiguous.
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
WIND_CSV = "data/eirgrid/ie_wind_hourly.csv"

SHEET = "System Data"
WANTED_COLUMNS = ("DateTime", "GMT Offset", "IE Wind Generation", "IE Wind Availability")
REQUIRED_COLUMNS = ("DateTime", "IE Wind Generation", "IE Wind Availability")  # GMT Offset optional


def to_hourly_utc(raw):
    """Quarter-hourly local-time rows -> hourly UTC rows (ie_wind_mw, ie_wind_avail_mw).

    An hour is kept only if all four of its quarter-hours have a wind reading."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"EirGrid data is missing columns {missing}; found {list(raw.columns)[:12]}...")

    d = raw.dropna(subset=["DateTime"]).copy()
    local = pd.to_datetime(d["DateTime"])
    if "GMT Offset" in d.columns:
        utc = local - pd.to_timedelta(d["GMT Offset"], unit="h")
    else:  # fall back to the Irish DST rules; 'infer' resolves the repeated autumn hour by order
        utc = pd.DatetimeIndex(local).tz_localize("Europe/Dublin", ambiguous="infer",
                                                  nonexistent="raise").tz_convert("UTC").tz_localize(None)
        utc = pd.Series(utc, index=d.index)

    q = pd.DataFrame({"ie_wind_mw": d["IE Wind Generation"].to_numpy(),
                      "ie_wind_avail_mw": d["IE Wind Availability"].to_numpy()},
                     index=pd.DatetimeIndex(utc.to_numpy(), name="time")).sort_index()
    if q.index.duplicated().any():
        raise ValueError("duplicate UTC timestamps: check the GMT Offset column / overlapping files")

    hourly = q.resample("1h").mean()
    complete = q["ie_wind_mw"].resample("1h").count() == 4
    return hourly[complete].dropna().reset_index()


def load_wind_hourly(path=None):
    """Read the cleaned hourly CSV written by scripts/ingest_eirgrid.py."""
    path = Path(path) if path else ROOT_DIR / WIND_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/ingest_eirgrid.py first.")
    df = pd.read_csv(path, parse_dates=["time"])
    if (df["ie_wind_mw"] < 0).any():
        raise ValueError("negative wind generation in the EirGrid data")
    return df