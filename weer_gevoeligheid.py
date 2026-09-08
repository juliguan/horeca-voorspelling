"""
Eerlijkheids-check op de backtest: hoe goed is het omzetmodel ECHT, als je
in de testperiode niet het (achteraf bekende) weer gebruikt, maar weer
zoals een voorspelling er op dat moment realistisch had uitgezien?

Waarom dit ertoe doet: model.py en app.py gebruiken voor de testperiode
archief-weer (fetch_historical_weather) -- exact wat er die dag geweest is.
Op het moment dat je in het echt een rooster zou maken, wist je dat nog
niet: je had een weersvoorspelling, met de onzekerheid die daarbij hoort.
Als het model gevoelig is voor afwijkingen in temp_c/neerslag_mm/zon_index,
overschat de bestaande backtest (met archief-weer) de nauwkeurigheid die je
in de praktijk zou halen.

Aanpak: zelfde train/test-split en zelfde model als model.py, maar twee
varianten van de testperiode:
  - archief:   temp_c/neerslag_mm/zon_index zoals het WERKELIJK is geweest
               (ongewijzigd t.o.v. de bestaande backtest in model.py)
  - voorspeld: dezelfde kolommen, maar zoals een 7-dagen-vooruit voorspelling
               er op dat moment uitzag (Open-Meteo's Previous Runs API, zie
               src/weather.py::fetch_forecast_weather)
Traint EENMAAL op train (archief-weer -- op trainmomenten ligt het weer al
lang vast, dat is geen lek, het model leert alleen een verband) en scoort
dezelfde testdagen tweemaal: met archief-testweer en met voorspeld-testweer.
Het verschil in MAPE is de eerlijke schatting van wat voorspel-onzekerheid
aan nauwkeurigheid kost.

Alleen zinvol voor een testperiode binnen het venster dat Open-Meteo's
Previous Runs API bewaart: vanaf ~begin 2024, tot 7 dagen vooruit per
gevraagde dag. Voor de synthetische data hier valt de testperiode (laatste
~20% van 2023-09 t/m 2026-08) daar ruim binnen.

Run: source .venv/bin/activate && python weer_gevoeligheid.py
"""
import numpy as np
import pandas as pd

from src.data_loading import load_kassa_orderregels
from src.aggregation import aggregate_orders_to_daily
from src.weather import geocode, fetch_historical_weather, fetch_forecast_weather
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split
from src.evaluation import scoreboard
import model

LOCATIE = "Cafe Laurierboom, Amsterdam"
WEER_KOLOMMEN = ["temp_c", "neerslag_mm", "zon_index"]


def _bouw_features(daily: pd.DataFrame, weer: pd.DataFrame) -> pd.DataFrame:
    df = daily.merge(weer, on="datum", how="left")
    df[WEER_KOLOMMEN] = df[WEER_KOLOMMEN].ffill()
    df["event"] = None  # los van dit onderzoek -- events zijn identiek in beide varianten
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    return df


def main():
    orders = load_kassa_orderregels()
    daily = aggregate_orders_to_daily(orders)

    loc = geocode(LOCATIE)
    start = daily["datum"].min().strftime("%Y-%m-%d")
    end = daily["datum"].max().strftime("%Y-%m-%d")

    dates = np.sort(daily["datum"].unique())
    test_start = pd.Timestamp(dates[int(len(dates) * 0.8)]).strftime("%Y-%m-%d")
    test_end = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")

    print(f"archief-weer ophalen voor {loc['naam']} ({start} t/m {end})...")
    archief = fetch_historical_weather(loc["lat"], loc["lon"], start, end)

    print(f"7-dagen-vooruit voorspelling ophalen voor de testperiode ({test_start} t/m {test_end})...")
    voorspeld_test = fetch_forecast_weather(loc["lat"], loc["lon"], test_start, test_end, lead_days=7)

    # variant 'archief': ongewijzigd, precies zoals de bestaande backtest in model.py.
    df_archief = _bouw_features(daily, archief)

    # variant 'voorspeld': in de testperiode het archief-weer vervangen door het
    # 7-dagen-vooruit voorspelde weer; in de trainperiode blijft archief-weer staan.
    weer_gemengd = archief.copy().set_index("datum")
    voorspeld_idx = voorspeld_test.set_index("datum")
    weer_gemengd.loc[voorspeld_idx.index, WEER_KOLOMMEN] = voorspeld_idx[WEER_KOLOMMEN]
    df_voorspeld = _bouw_features(daily, weer_gemengd.reset_index())

    train, test_archief = time_split(df_archief, test_start)
    _, test_voorspeld = time_split(df_voorspeld, test_start)

    # alleen open dagen (omzet > 0) -- MAPE is ongedefinieerd bij omzet 0,
    # zelfde aanname als de rest van de schil (src/evaluation.py).
    open_mask = (test_archief[model.TARGET] > 0).to_numpy()
    test_archief, test_voorspeld = test_archief[open_mask], test_voorspeld[open_mask]

    y_test = test_archief[model.TARGET].to_numpy()
    y_pred_archief = model.predict(train, test_archief, model.TARGET)
    y_pred_voorspeld = model.predict(train, test_voorspeld, model.TARGET)

    print()
    print(f"testperiode: {test_start} t/m {test_end} ({len(y_test)} open dagen)")
    print(f"gem. afwijking temp_c tussen archief en voorspeld: "
          f"{(test_archief['temp_c'].to_numpy() - test_voorspeld['temp_c'].to_numpy()).mean():+.2f}°C "
          f"(gem. |afwijking| {np.abs(test_archief['temp_c'].to_numpy() - test_voorspeld['temp_c'].to_numpy()).mean():.2f}°C)")
    print()

    resultaten = {
        "backtest met archief-weer (huidige aanpak, optimistisch)": (y_test, y_pred_archief),
        "backtest met 7-dagen-vooruit voorspeld weer (realistisch)": (y_test, y_pred_voorspeld),
    }
    print(scoreboard(resultaten).to_string(index=False))


if __name__ == "__main__":
    main()
