"""
Test het gekozen model (model.py, ONGEWIJZIGD) en de lineaire baseline
(src/baselines.py, ONGEWIJZIGD) op 5 onafhankelijk gegenereerde datasets van
Astra (astra_data/*.py, zie astra_data/LEESMIJ.md) -- vijf verschillende
soorten horeca-zaken, elk met een eigen vraagproces waar wij geen inzage in
hadden bij het schrijven van model.py.

Belangrijk verschil met de synthetische dataset in data/: deze CSV's bevatten
BEWUST geen weer- of event-kolommen (zie LEESMIJ.md, "Kalenderdagen en
ontbrekende data"). We verzinnen die dus niet zelf bij (echt Amsterdams weer
zou geen enkel verband hebben met de fictieve vraag van deze fictieve
zaken) -- we vullen ze in als constante, dus informatieloze kolommen, zodat
model.py's vaste FEATURE_COLS nog steeds bestaan maar het model er terecht
niks aan heeft. Dit is dus een eerlijke test van alleen het
kalender+lag/rolling-deel van de aanpak, niet van het weer/event-deel.

Rapporteert alle 5 uitkomsten naast elkaar -- niet de gunstigste eruit
pikken.

Run: source .venv/bin/activate && python evalueer_astra_datasets.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregation import aggregate_orders_to_daily
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split
from src.evaluation import scoreboard
from src.baselines import predict_weekday_average, predict_same_weekday_n_weeks_ago, predict_simple_linear
import model

ASTRA_DIR = Path(__file__).resolve().parent / "astra_data"
DATASETS = ["buurtcafe", "kantoorlunchroom", "hotelrestaurant", "grandcafe", "avondrestaurant"]


def bouw_features(pad: Path) -> pd.DataFrame:
    orders = pd.read_csv(pad, parse_dates=["tijdstip"])
    daily = aggregate_orders_to_daily(orders)

    # geen weer/events in deze dataset (met opzet) -- constant, dus geen
    # informatie, in plaats van zelf iets verzinnen dat er niet bij hoort.
    daily["temp_c"] = 0.0
    daily["neerslag_mm"] = 0.0
    daily["zon_index"] = 0.0
    daily["event"] = None

    df = add_calendar_features(daily)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    return df


def evalueer_dataset(naam: str) -> pd.DataFrame:
    df = bouw_features(ASTRA_DIR / f"{naam}.csv")

    dates = np.sort(df["datum"].unique())
    test_start = pd.Timestamp(dates[int(len(dates) * 0.8)]).strftime("%Y-%m-%d")
    train, test = time_split(df, test_start)

    # net als model.py/src/evaluation.py: MAPE is ongedefinieerd bij omzet 0
    # (gesloten dagen), dus alleen open dagen meenemen.
    open_mask = (test[model.TARGET] > 0).to_numpy()
    test = test[open_mask].reset_index(drop=True)
    y_test = test[model.TARGET]

    resultaten = {
        "HistGradientBoostingRegressor (log-omzet) -- huidige keuze": (y_test, model.predict(train, test, model.TARGET)),
        "lineair weekdag+maand+weer+event+trend (baseline)": (y_test, predict_simple_linear(train, test, model.TARGET)),
        "zelfde dag 4 weken terug (baseline)": (y_test, predict_same_weekday_n_weeks_ago(df, test["datum"], model.TARGET)),
        "weekdaggemiddelde (baseline)": (y_test, predict_weekday_average(train, test, model.TARGET)),
    }
    bord = scoreboard(resultaten)
    bord.insert(0, "dataset", naam)
    bord.insert(1, "test_start", test_start)
    bord.insert(2, "n_test_dagen", len(y_test))
    return bord


def main():
    alle = pd.concat([evalueer_dataset(naam) for naam in DATASETS], ignore_index=True)
    pd.set_option("display.width", 160)
    for naam in DATASETS:
        print(f"\n=== {naam} ===")
        subset = alle[alle["dataset"] == naam].drop(columns=["dataset"])
        print(subset.to_string(index=False))

    print("\n\n=== samenvatting: waar staat het huidige model (HGB) t.o.v. de lineaire baseline? ===")
    hgb = alle[alle["model"].str.startswith("HistGradientBoosting")].set_index("dataset")["mape"]
    lin = alle[alle["model"].str.startswith("lineair")].set_index("dataset")["mape"]
    samenvatting = pd.DataFrame({"MAPE HGB (huidig)": hgb, "MAPE lineair (baseline)": lin})
    samenvatting["HGB wint"] = samenvatting["MAPE HGB (huidig)"] < samenvatting["MAPE lineair (baseline)"]
    print(samenvatting.round(2).to_string())


if __name__ == "__main__":
    main()
