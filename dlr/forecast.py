
import math
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

# The target is ampacity in AMPS. It used to be headroom % over the seasonal static rating, which
# jumped at every season boundary even when the weather did not change. Headroom is derived
# afterwards (ampacity / static_rating - 1) for reporting only.
TARGET = "dlr_ampacity"

FEATURES = [
    "lead_h", "hour", "doy_sin", "doy_cos",                      # what is being predicted
    "v_now", "t_now", "wdir_sin", "wdir_cos",                    # state at issue time
    "v_mean24", "v_std24", "t_mean24", "t_std24",                # last 24 h up to issue time
    "amp_now", "amp_mean24", "amp_lag48",                        # recent ampacity (always known)
]


def build_dataset(df, issue_hour=config.FORECAST_ISSUE_HOUR_UTC):
    """One row per target hour, with features known at that hour's issue time.

    df is add_dlr_columns() output. Returns the rows that have a complete feature set,
    with columns time, issue_time, season, static_rating, hour, dlr_ampacity (the target, A)
    and FEATURES."""
    g = df.set_index("time").sort_index()
    if g.index.has_duplicates:
        raise ValueError("duplicate timestamps in weather data")
    g = g[["t_ambient", "v_wind", "wind_dir", TARGET]].reindex(
        pd.date_range(g.index.min(), g.index.max(), freq="h"))      # gaps become NaN, rows drop later

    state = pd.DataFrame(index=g.index)
    state["v_now"] = g["v_wind"]
    state["t_now"] = g["t_ambient"]
    rad = np.deg2rad(g["wind_dir"])
    state["wdir_sin"], state["wdir_cos"] = np.sin(rad), np.cos(rad)
    for name, col in (("v", "v_wind"), ("t", "t_ambient")):
        win = g[col].rolling(24, min_periods=24)                    # window ends at the row's own hour
        state[f"{name}_mean24"], state[f"{name}_std24"] = win.mean(), win.std()
    state["amp_now"] = g[TARGET]
    state["amp_mean24"] = g[TARGET].rolling(24, min_periods=24).mean()

    out = df[["time", "season", "static_rating", TARGET]].reset_index(drop=True)
    out["issue_time"] = out["time"].dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=issue_hour)
    out["lead_h"] = (out["time"] - out["issue_time"]) / pd.Timedelta(hours=1)
    feat = state.reindex(pd.DatetimeIndex(out["issue_time"]))       # state AT the issue time
    feat.index = out.index
    out = pd.concat([out, feat], axis=1)
    # t-48 h is always before the issue time (lead is at most 47 - issue_hour = 35 h)
    out["amp_lag48"] = g[TARGET].shift(48).reindex(pd.DatetimeIndex(out["time"])).to_numpy()
    out["hour"] = out["time"].dt.hour
    angle = 2 * np.pi * out["time"].dt.dayofyear / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(angle), np.cos(angle)
    return out.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def blocked_split(times, fractions=config.SPLIT_FRACTIONS, block_days=config.SPLIT_BLOCK_DAYS,
                  gap_days=config.SPLIT_GAP_DAYS, seed=config.SPLIT_SEED):

    day = (pd.DatetimeIndex(times).normalize() - pd.DatetimeIndex(times).normalize().min()).days.to_numpy()
    block, pos = day // block_days, day % block_days
    n_blocks = block.max() + 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    cuts = np.cumsum(np.round(np.array(fractions) * n_blocks).astype(int))
    set_of_block = np.empty(n_blocks, dtype=int)
    for s, (lo, hi) in enumerate(zip([0, *cuts[:-1]], [*cuts[:-1], n_blocks])):
        set_of_block[order[lo:hi]] = s
    own = set_of_block[block]
    prev = np.where(block > 0, set_of_block[np.maximum(block - 1, 0)], own)
    keep = ~((pos < gap_days) & (prev != own))
    return tuple(keep & (own == s) for s in range(3))


def fit_quantile_models(X, y, quantiles=config.FORECAST_QUANTILES, params=config.LGBM_PARAMS):
    return {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(X, y) for q in quantiles}


def predict_quantiles(models, X):
    """DataFrame, one column per quantile. Rows are sorted so that a higher quantile is never
    below a lower one (independently trained quantile models can cross)."""
    qs = sorted(models)
    raw = np.column_stack([models[q].predict(X) for q in qs])
    return pd.DataFrame(np.sort(raw, axis=1), columns=qs)


def conformal_shift(q, pred_cal, y_cal):
    """Shift to add to quantile-q predictions so they hold on new data (split conformal).

    Uses a held-out calibration set the model was NOT trained on. For a lower quantile
    (q < 0.5) the shifted forecast is exceeded by the actual value with probability at least
    1 - q; for an upper quantile, the actual stays below it with probability at least q.
    The guarantee assumes calibration and test data are exchangeable, which blocked random
    splitting approximates but time series do not strictly satisfy. The median is left alone."""
    pred_cal, y_cal = np.asarray(pred_cal, float), np.asarray(y_cal, float)
    n = len(y_cal)
    if q == 0.5:
        return 0.0
    if q < 0.5:
        k = math.ceil((n + 1) * (1 - q))
        scores = np.sort(pred_cal - y_cal)
        sign = -1.0
    else:
        k = math.ceil((n + 1) * q)
        scores = np.sort(y_cal - pred_cal)
        sign = 1.0
    if k > n:
        raise ValueError(f"calibration set of {n} is too small for quantile {q}")
    return sign * scores[k - 1]


def calibrate(pred_cal, y_cal, pred_test):
    """Apply conformal_shift to every column; returns the adjusted test predictions."""
    out = pred_test.copy()
    for q in pred_test.columns:
        out[q] = pred_test[q] + conformal_shift(q, pred_cal[q], y_cal)
    return pd.DataFrame(np.sort(out.to_numpy(), axis=1), columns=out.columns)


def pinball_loss(y, pred, q):
    d = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def climatology_quantiles(train, target, quantiles):
    """Baseline: quantiles of the TRAINING ampacity for the same hour of day and season
    (hour of day alone if a season has no training rows)."""
    keys = ["hour", "season"]
    out = pd.DataFrame(index=range(len(target)), columns=list(quantiles), dtype=float)
    idx = pd.MultiIndex.from_frame(target[keys])
    for q in quantiles:
        by_key = train.groupby(keys)[TARGET].quantile(q).reindex(idx).to_numpy()
        by_hour = target["hour"].map(train.groupby("hour")[TARGET].quantile(q)).to_numpy()
        out[q] = np.where(np.isnan(by_key), by_hour, by_key)
    return pd.DataFrame(np.sort(out.to_numpy(), axis=1), columns=out.columns)


def day_block_bootstrap(flags, times, n_boot=2000, seed=0):
    """95% interval for the mean of 0/1 `flags`, resampling whole days. Hourly rows within a
    day are strongly dependent, so a plain binomial interval would be far too narrow."""
    flags = np.asarray(flags, float)
    days = pd.DatetimeIndex(times).normalize()
    per_day = pd.DataFrame({"f": flags, "d": days}).groupby("d")["f"].agg(["sum", "count"])
    s, c = per_day["sum"].to_numpy(), per_day["count"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(s), size=(n_boot, len(s)))
    means = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))