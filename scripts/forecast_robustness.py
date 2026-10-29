import pandas as pd
from dlr import config
from dlr.forecast import (FEATURES, TARGET, blocked_split, build_dataset, calibrate, climatology_quantiles,
                          fit_quantile_models, pinball_loss, predict_quantiles)
from dlr.pipeline import add_dlr_columns, load_weather

N_SPLITS = 8
RISK = config.FORECAST_RISK_QUANTILES


def run():
    ds = build_dataset(add_dlr_columns(load_weather()))
    rows = []
    for seed in range(N_SPLITS):
        train, cal, test = (ds[m].reset_index(drop=True) for m in blocked_split(ds["time"], seed=seed))
        models = fit_quantile_models(train[FEATURES], train[TARGET])
        raw = predict_quantiles(models, test[FEATURES])
        conf = calibrate(predict_quantiles(models, cal[FEATURES]), cal[TARGET].to_numpy(), raw)
        clim = climatology_quantiles(train, test, config.FORECAST_QUANTILES)
        y = test[TARGET].to_numpy()
        row = {"seed": seed}
        for q in (*RISK, 0.5):
            row[f"skill_P{round(q * 100)}_%"] = 100 * (1 - pinball_loss(y, conf[q], q) / pinball_loss(y, clim[q], q))
        for q in RISK:
            row[f"below_P{round(q * 100)}_model+CQR"] = (y < conf[q].to_numpy()).mean()
            row[f"below_P{round(q * 100)}_raw"] = (y < raw[q].to_numpy()).mean()
            row[f"below_P{round(q * 100)}_clim"] = (y < clim[q].to_numpy()).mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("seed")
    print(f"Skill vs climatology (positive = better) and share of hours below the forecast, "
          f"{N_SPLITS} random block assignments:\n")
    print(table.round(3).to_string())
    print("\nMean:\n" + table.mean().round(3).to_string())
    print("\nMin / max:\n" + pd.DataFrame({"min": table.min(), "max": table.max()}).round(3).to_string())
    print("\nNominal below-share: " + ", ".join(f"P{round(q * 100)} = {q:.2f}" for q in RISK))


if __name__ == "__main__":
    run()