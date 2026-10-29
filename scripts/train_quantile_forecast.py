
import matplotlib.pyplot as plt
import numpy as np
from dlr import config
from dlr.forecast import (FEATURES, TARGET, blocked_split, build_dataset, calibrate, climatology_quantiles,
                          day_block_bootstrap, fit_quantile_models, pinball_loss, predict_quantiles)
from dlr.pipeline import ROOT_DIR, add_dlr_columns, load_weather

Q = config.FORECAST_QUANTILES


def run():
    ds = build_dataset(add_dlr_columns(load_weather()))
    train_m, cal_m, test_m = blocked_split(ds["time"])
    train, cal, test = (ds[m].reset_index(drop=True) for m in (train_m, cal_m, test_m))
    print(f"Rows: train {len(train)} | calibration {len(cal)} | test {len(test)} "
          f"({test['time'].dt.normalize().nunique()} test days)")
    if test["time"].dt.normalize().nunique() < 30:
        print("WARNING: under 30 test days. Coverage estimates below are very noisy.")

    models = fit_quantile_models(train[FEATURES], train[TARGET])
    raw_cal = predict_quantiles(models, cal[FEATURES])
    raw = predict_quantiles(models, test[FEATURES])
    conf = calibrate(raw_cal, cal[TARGET].to_numpy(), raw)       # shifts fitted on cal only
    clim = climatology_quantiles(train, test, Q)
    y = test[TARGET].to_numpy()

    print("\nPinball loss in amps on the test set (lower is better; skill = improvement over climatology):")
    print(f"{'quantile':>9} {'model':>8} {'model+CQR':>10} {'climatology':>12} {'skill (raw)':>12} {'skill (CQR)':>12}")
    for q in Q:
        pr, pc, pk = (pinball_loss(y, f[q], q) for f in (raw, conf, clim))
        print(f"{q:>9.2f} {pr:>8.2f} {pc:>10.2f} {pk:>12.2f} {100 * (1 - pr / pk):>11.1f}% {100 * (1 - pc / pk):>11.1f}%")

    print("\nDownside risk: share of test hours where actual ampacity was BELOW the forecast "
          "(should equal the quantile; 95% interval resamples whole days):")
    print(f"{'quantile':>9} {'model':>8} {'model+CQR':>10} {'CQR 95% interval':>20} {'climatology':>12}")
    for q in config.FORECAST_RISK_QUANTILES:
        below = lambda f: (y < f[q].to_numpy())
        lo, hi = day_block_bootstrap(below(conf), test["time"])
        print(f"{q:>9.2f} {below(raw).mean():>8.3f} {below(conf).mean():>10.3f} "
              f"{lo:>10.3f} - {hi:<7.3f} {below(clim).mean():>12.3f}")
    r = config.FORECAST_RISK_QUANTILES[0]
    static = test["static_rating"].to_numpy()
    print(f"\nMean forecast P{int(r * 100)} ampacity {conf[r].mean():.0f} A vs mean actual {y.mean():.0f} A "
          f"(P{int(r * 100)} is {np.mean(conf[r].to_numpy() / static - 1) * 100:.0f}% over the seasonal static rating "
          f"on average): the usable margin is the low quantile, not the average.")

    mae = lambda p: np.mean(np.abs(y - np.asarray(p)))
    print("\nMedian forecast, mean absolute error (A):")
    print(f"  model {mae(raw[0.5]):.0f} | climatology {mae(clim[0.5]):.0f} | "
          f"48 h persistence {mae(test['amp_lag48']):.0f}")

    nominal = np.array(Q)
    plt.figure(figsize=(5, 5), dpi=200)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, f, style in (("model", raw, "o-"), ("model + CQR", conf, "s-"), ("climatology", clim, "^:")):
        plt.plot(nominal, [(y <= f[q].to_numpy()).mean() for q in Q], style, label=name)
    plt.xlabel("Nominal quantile")
    plt.ylabel("Observed share of hours at or below forecast")
    plt.title("Reliability (test set)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out = ROOT_DIR / "figures" / "forecast_reliability.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out)
    print(f"\nSaved {out}")
    print("\nReminder: inputs are ERA5 observations, not weather forecasts; see dlr/forecast.py.")


if __name__ == "__main__":
    run()