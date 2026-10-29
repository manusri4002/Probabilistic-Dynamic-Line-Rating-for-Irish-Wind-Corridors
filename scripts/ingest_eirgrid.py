"""Convert EirGrid 'System Data Qtr Hourly' workbooks into one clean hourly UTC wind CSV.

Usage:
    python scripts/ingest_eirgrid.py                      # every data/eirgrid/System-Data-Qtr-Hourly-*.xlsx
    python scripts/ingest_eirgrid.py path/to/file.xlsx    # or name files / wildcards explicitly
"""

import glob
import sys

import pandas as pd

from dlr.eirgrid import ROOT_DIR, SHEET, WANTED_COLUMNS, WIND_CSV, to_hourly_utc


def main(args):
    patterns = args or [str(ROOT_DIR / "data" / "eirgrid" / "System-Data-Qtr-Hourly-*.xlsx")]
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No workbooks found for {patterns}")

    frames = []
    for path in paths:
        raw = pd.read_excel(path, sheet_name=SHEET, usecols=lambda c: c in WANTED_COLUMNS)
        print(f"Read {len(raw):>6} rows from {path}")
        frames.append(raw)

    raw = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("DateTime", "GMT Offset") if c in raw.columns]
    raw = raw.drop_duplicates(subset=keys)

    hourly = to_hourly_utc(raw)
    out = ROOT_DIR / WIND_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out, index=False)
    print(f"Saved {len(hourly)} hourly UTC rows ({hourly['time'].min()} to {hourly['time'].max()}) to {out}")
    print(f"Mean ROI wind output {hourly['ie_wind_mw'].mean():.0f} MW, max {hourly['ie_wind_mw'].max():.0f} MW")


if __name__ == "__main__":
    main(sys.argv[1:])