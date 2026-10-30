"""Headroom as a band, not a single number.

Three inputs are assumptions rather than measurements:
  * the line bearing (one ERA5 point stands in for a whole corridor),
  * the sheltering factor (how much slower the wind at the conductor is than the grid-box wind),
  * the sky-diffuse model (CIGRE TB 601 versus the isotropic TB 207 form).
Each scenario is one combination of the three. The report quotes the WORST span (the bearing with
the lowest P5 headroom) and the range of its results across the plausible sheltering and
diffuse-model values, so no single assumed value is carried forward as a result.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from . import config
from .pipeline import add_dlr_columns, solar_on_conductor

METRICS = ("median_pct", "p5_pct", "p1_pct", "below_static_h", "below_static_pct")
SCENARIO_KEYS = ["sheltering", "diffuse_model"]


def scenario_table(df, bearings=config.BEARING_SWEEP, shelterings=config.SHELTERING_SWEEP,
                   diffuse_models=config.DIFFUSE_MODELS, albedo=config.GROUND_ALBEDO,
                   conductor=config.DRAKE, line=config.LINE):
    """One row per (bearing, sheltering, diffuse model): headroom over the seasonal static rating.

    median_pct / p5_pct / p1_pct: headroom exceeded in 50 / 95 / 99 % of hours.
    below_static_h: hours when the dynamic rating is LOWER than the static one."""
    rows = []
    for bearing in bearings:
        scenario_line = replace(line, bearing_deg=float(bearing))
        for model in diffuse_models:
            # sun geometry depends on bearing and diffuse model only, so compute it once and reuse it
            # for every sheltering factor (the geometric solar terms are then passed in as solar_rad)
            solar = solar_on_conductor(df, scenario_line, "auto", albedo, model)
            base = df.assign(solar_rad=solar)
            for sheltering in shelterings:
                ratio = add_dlr_columns(base, conductor=conductor, line=scenario_line,
                                        sheltering=sheltering, solar_model="ghi")["headroom_ratio"].to_numpy()
                rows.append({
                    "bearing_deg": float(bearing), "sheltering": float(sheltering), "diffuse_model": model,
                    "median_pct": (np.median(ratio) - 1.0) * 100.0,
                    "p5_pct": (np.quantile(ratio, 0.05) - 1.0) * 100.0,
                    "p1_pct": (np.quantile(ratio, 0.01) - 1.0) * 100.0,
                    "below_static_h": int((ratio < 1.0).sum()),
                    "below_static_pct": float((ratio < 1.0).mean() * 100.0),
                })
    return pd.DataFrame(rows)


def worst_span(table):
    """For each (sheltering, diffuse model) the bearing with the lowest P5 headroom, with its statistics.

    P5 is the number an operator can rely on, so it decides which bearing counts as worst. When the
    P5 hours are near-calm, natural convection sets the rating whatever the wind angle and every
    bearing ties on P5; the tie is then broken by the lower median headroom, and after that by
    the smaller bearing."""
    ordered = table.sort_values(["p5_pct", "median_pct", "bearing_deg"])
    worst = ordered.groupby(SCENARIO_KEYS, sort=False).head(1)
    return worst.sort_values(["diffuse_model", "sheltering"], ascending=[True, False]).reset_index(drop=True)


def band(worst):
    """Lowest and highest value of every metric over the worst-span rows."""
    return worst[list(METRICS)].agg(["min", "max"]).T


def bearing_profile(table, sheltering=config.SHELTERING_FACTOR, diffuse_model=config.DIFFUSE_MODEL):
    """Headroom statistics by bearing for one sheltering factor and diffuse model."""
    sel = table[(np.isclose(table["sheltering"], sheltering)) & (table["diffuse_model"] == diffuse_model)]
    return sel.set_index("bearing_deg")[list(METRICS)]
