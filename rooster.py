"""
Roosteradvies per dagdeel (ochtend/lunch/middag/diner/avond) -- specifieker
dan staffing.py, dat alleen een dagtotaal geeft. Los bestand, hergebruikt
model.py (de omzetvoorspelling) en staffing.py (de norm-omzet-per-uur),
geen van beide bestanden aangepast.

Aanpak:
  1. Uit kassa_orderregels.csv: het historische aandeel van elk dagdeel in
     de dagomzet, per weekdag -- berekend op de trainperiode, dus geen
     lekkage uit de testperiode.
  2. De voorspelde dagomzet komt uit model.py (hergebruikt, niet opnieuw
     getraind); dat aandeel verdeelt 'm over de dagdelen.
  3. Omzet per dagdeel -> uren via dezelfde norm als staffing.py (omzet per
     gewerkt uur) -> mensen via een aangenomen gemiddelde shiftlengte,
     naar boven afgerond (je kan geen halve medewerker inroosteren).

LET OP: dagstaat.csv heeft alleen een dagtotaal aan ingeroosterde uren, geen
uitsplitsing per dagdeel. Er is dus geen "werkelijk aantal mensen per
dagdeel" om tegenaan te leggen -- dit is een advies, geen scoreboard zoals
bij de andere secties.

Run: source .venv/bin/activate && python rooster.py
"""
import numpy as np
import pandas as pd

from src.data_loading import load_dagstaat, load_kassa_orderregels
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split, TEST_START
from staffing import bepaal_norm_omzet_per_uur
import model

DAGDELEN = ["ochtend", "lunch", "middag", "diner", "avond"]
GEMIDDELDE_SHIFTLENGTE_UUR = 5.0  # aanname -- pas aan naar je eigen shiftlengte


def dagdeel_aandeel_per_weekday(kassa: pd.DataFrame, train_end: str = TEST_START) -> pd.DataFrame:
    """Historisch aandeel van elk dagdeel in de dagomzet, per weekdag.
    Alleen train-periode (voor train_end), dus geen lekkage uit de testperiode."""
    k = kassa[kassa["datum"] < train_end]
    per_dag_dagdeel = k.groupby(["datum", "dagdeel"], observed=True)["regelbedrag"].sum().reset_index()
    per_dag_dagdeel["weekday_num"] = per_dag_dagdeel["datum"].dt.weekday
    dagtotaal = per_dag_dagdeel.groupby("datum")["regelbedrag"].transform("sum")
    per_dag_dagdeel["aandeel"] = per_dag_dagdeel["regelbedrag"] / dagtotaal

    aandeel = (
        per_dag_dagdeel.groupby(["weekday_num", "dagdeel"], observed=True)["aandeel"]
        .mean()
        .unstack("dagdeel")
        .reindex(columns=DAGDELEN)
        .fillna(0.0)
    )
    # normaliseer zodat elke weekdagrij optelt tot 1 (voor het geval een
    # dagdeel structureel ontbreekt op sommige dagen van die weekdag)
    return aandeel.div(aandeel.sum(axis=1), axis=0)


def roosteradvies(
    voorspelde_dagomzet: np.ndarray,
    datums: pd.Series,
    aandeel: pd.DataFrame,
    norm_omzet_per_uur: float,
    shiftlengte_uur: float = GEMIDDELDE_SHIFTLENGTE_UUR,
) -> pd.DataFrame:
    """Voorspelde dagomzet -> omzet per dagdeel -> uren -> mensen (naar boven afgerond)."""
    weekday_num = datums.dt.weekday.to_numpy()
    aandeel_per_dag = aandeel.loc[weekday_num].to_numpy()  # (n_dagen, n_dagdelen)
    omzet_per_dagdeel = np.asarray(voorspelde_dagomzet, dtype=float)[:, None] * aandeel_per_dag

    uren_per_dagdeel = omzet_per_dagdeel / norm_omzet_per_uur
    mensen_per_dagdeel = np.ceil(uren_per_dagdeel / shiftlengte_uur).astype(int)

    result = pd.DataFrame(mensen_per_dagdeel, columns=DAGDELEN)
    result.insert(0, "datum", datums.to_numpy())
    return result


def main():
    kassa = load_kassa_orderregels()
    aandeel = dagdeel_aandeel_per_weekday(kassa, TEST_START)
    print("gemiddeld aandeel in dagomzet per dagdeel, per weekdag (trainperiode):")
    print((aandeel * 100).round(1).to_string())
    print()

    df = load_dagstaat()
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    train, test = time_split(df, TEST_START)

    norm = bepaal_norm_omzet_per_uur(train)
    omzet_pred = model.predict(train, test, model.TARGET)

    advies = roosteradvies(omzet_pred, test["datum"].reset_index(drop=True), aandeel, norm)
    print("roosteradvies (aantal mensen per dagdeel), eerste 10 dagen van de testperiode:")
    print(advies.head(10).to_string(index=False))
    print()
    print("gemiddeld aantal mensen per dagdeel over de hele testperiode:")
    print(advies[DAGDELEN].mean().round(1).to_string())


if __name__ == "__main__":
    main()
