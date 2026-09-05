"""
HIER komt jouw model. De schil (data_loading, features, split, evaluation,
baselines) staat in src/ en run_baselines.py -- die regelt inlezen, features
bouwen, chronologisch splitsen en scoren. Modelkeuze en featureselectie zijn
aan jou.

Huidige keuze: HistGradientBoostingRegressor op log(omzet), standaardparameters,
met de kalender/weer/event/lag/rolling-features die al in features.py staan.

Run: source .venv/bin/activate && python model.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_loading import load_dagstaat
from src.features import (
    add_calendar_features,
    add_event_flags,
    add_weather_features,
    add_lag_features,
    add_rolling_features,
)
from src.split import time_split, TEST_START
from src.evaluation import scoreboard
from src.baselines import predict_weekday_average, predict_same_weekday_n_weeks_ago, predict_simple_linear

TARGET = "omzet"

# kalender (weekdag, maand, trend), weer, event-vlag, lags, rolling -- allemaal
# al aanwezig na add_calendar_features/add_event_flags/add_weather_features/
# add_lag_features/add_rolling_features in main(). Niks nieuws toegevoegd.
FEATURE_COLS = [
    "weekday_num", "month", "trend", "day_of_year",
    "temp_c", "neerslag_mm", "zon_index", "is_droog",
    "is_event",
    "omzet_lag_7", "omzet_lag_14", "omzet_lag_21", "omzet_lag_28",
    "omzet_rollmean_7", "omzet_rollstd_7",
    "omzet_rollmean_28", "omzet_rollstd_28",
]


def predict(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """HistGradientBoostingRegressor, getraind op log(target), standaardparameters
    (geen tuning). Voorspelling wordt met exp() teruggezet naar de omzet-schaal."""
    # log(0) is ongedefinieerd -- de enkele gesloten dagen (omzet=0) in train
    # horen niet mee in een log-getraind model.
    train_fit = train[train[target] > 0]

    X_train = train_fit[FEATURE_COLS]
    y_train = np.log(train_fit[target])
    X_test = test[FEATURE_COLS]

    reg = HistGradientBoostingRegressor(random_state=0, categorical_features=["weekday_num", "month"])
    reg.fit(X_train, y_train)

    log_pred = reg.predict(X_test)
    return np.exp(log_pred)


def main():
    df = load_dagstaat()
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=TARGET)
    df = add_rolling_features(df, target=TARGET)

    train, test = time_split(df, TEST_START)
    y_test = test[TARGET]

    y_pred = predict(train, test, TARGET)

    results = {
        "HistGradientBoostingRegressor (log-omzet)": (y_test, y_pred),
        "weekdaggemiddelde (baseline)": (y_test, predict_weekday_average(train, test, TARGET)),
        "zelfde dag 4 weken terug (baseline)": (y_test, predict_same_weekday_n_weeks_ago(df, test["datum"], TARGET)),
        "lineair weekdag+maand+weer+event+trend (baseline)": (y_test, predict_simple_linear(train, test, TARGET)),
    }
    print(scoreboard(results).to_string(index=False))


if __name__ == "__main__":
    main()
