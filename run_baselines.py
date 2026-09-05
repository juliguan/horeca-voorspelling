"""
Sanity check op de schil: laadt data, bouwt features, splitst op datum,
en scoort de drie bekende baselines uit de opdracht. Doel is puur checken
dat data_loading/features/split/evaluation kloppen -- dit is niet het model.

Run: source .venv/bin/activate && python run_baselines.py
"""
import pandas as pd

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
from src.baselines import (
    predict_weekday_average,
    predict_same_weekday_n_weeks_ago,
    predict_simple_linear,
)
from model import predict as predict_gbm

TARGET = "omzet"

REFERENCE_MAPE = {
    "weekdaggemiddelde": 27,
    "zelfde dag 4 weken terug": 22,
    "weekdag+maand+weer+event+trend (lineair)": 12,
    "theoretische bodem (ruis)": 9,
}


def main():
    df = load_dagstaat()
    df = add_calendar_features(df)
    # extra features hieronder zijn alleen voor het GBM-model in model.py --
    # de baseline-functies gebruiken enkel de kolommen uit dagstaat.csv zelf.
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=TARGET)
    df = add_rolling_features(df, target=TARGET)

    train, test = time_split(df, TEST_START)
    print(f"train: {len(train)} dagen ({train.datum.min().date()} t/m {train.datum.max().date()})")
    print(f"test:  {len(test)} dagen ({test.datum.min().date()} t/m {test.datum.max().date()})")
    print()

    y_test = test[TARGET]

    pred_weekday_avg = predict_weekday_average(train, test, TARGET)
    pred_same_weekday = predict_same_weekday_n_weeks_ago(df, test["datum"], TARGET, n_weeks=4)
    pred_linear = predict_simple_linear(train, test, TARGET)
    pred_gbm = predict_gbm(train, test, TARGET)

    results = {
        "weekdaggemiddelde": (y_test, pred_weekday_avg),
        "zelfde dag 4 weken terug": (y_test, pred_same_weekday),
        "weekdag+maand+weer+event+trend (lineair)": (y_test, pred_linear),
        "HistGradientBoostingRegressor (log-omzet)": (y_test, pred_gbm),
    }

    board = scoreboard(results)
    print(board.to_string(index=False))
    print()
    print("Referentiewaarden uit de opdracht (op deze target/split):")
    for name, target_mape in REFERENCE_MAPE.items():
        actual = board.loc[board.model == name, "mape"]
        actual_str = f"{actual.iloc[0]:.1f}%" if len(actual) else "n.v.t. (geen baseline, is de vloer)"
        print(f"  {name}: verwacht ~{target_mape}% MAPE, gemeten {actual_str}")


if __name__ == "__main__":
    main()
