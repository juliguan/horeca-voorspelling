"""
Roosteradvies per dagdeel (ochtend/lunch/middag/diner/avond) -- specifieker
dan staffing.py, dat alleen een dagtotaal geeft. Los bestand, hergebruikt
model.py (de omzetvoorspelling) en staffing.py (de norm-omzet-per-uur),
geen van beide bestanden aangepast.

Aanpak (v2 -- zie onderaan waarom v1 tekortschoot):
  1. Per dagdeel een eigen model dat het AANDEEL in de dagomzet voorspelt
     uit dezelfde features als het omzetmodel (weekdag/maand/trend/weer/
     event) -- getraind op de trainperiode, dus geen lekkage. Voorspellingen
     worden genormaliseerd zodat ze per dag optellen tot 1.
  2. De voorspelde dagomzet komt uit model.py (hergebruikt, niet opnieuw
     getraind); dat aandeel verdeelt 'm over de dagdelen.
  3. Omzet per dagdeel -> uren via dezelfde norm als staffing.py (omzet per
     gewerkt uur) -> mensen via een aangenomen gemiddelde shiftlengte,
     naar boven afgerond (je kan geen halve medewerker inroosteren).

Waarom v2: v1 gebruikte een vast historisch aandeel per weekdag ("woensdag
is altijd ~25% lunch"), wat nooit reageert op de omstandigheden van een
specifieke dag -- regen raakt vooral een terraslunch, een avondevenement
raakt vooral het diner, maar een vast weekdaggemiddelde ziet dat verschil
niet. v2 laat de dagdeel-verdeling meebewegen met dezelfde signalen als de
dagvoorspelling zelf.

Dit dagdeel-aandeel is, in tegenstelling tot de rest van het roosteradvies,
wel te valideren: de werkelijke historische verdeling per dagdeel is
gewoon te berekenen uit de orderregels (zie evalueer_dagdeel_aanpak),
i.p.v. "er is geen werkelijk aantal mensen per dagdeel om tegenaan te
leggen" zoals bij het eindadvies (mensen per dagdeel) wel het geval blijft.

Run: source .venv/bin/activate && python rooster.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_loading import load_dagstaat, load_kassa_orderregels
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split, TEST_START
from staffing import bepaal_norm_omzet_per_uur
import model

DAGDELEN = ["ochtend", "lunch", "middag", "diner", "avond"]
GEMIDDELDE_SHIFTLENGTE_UUR = 5.0  # aanname -- pas aan naar je eigen shiftlengte

DAGDEEL_FEATURE_COLS = [
    "weekday_num", "month", "trend", "day_of_year",
    "temp_c", "neerslag_mm", "zon_index", "is_droog", "is_event",
]


def _dagdeel_aandelen_per_dag(kassa: pd.DataFrame) -> pd.DataFrame:
    """Werkelijk aandeel van elk dagdeel in de dagomzet, PER DAG (niet per
    weekdaggemiddelde). Dit is zowel het trainingsdoel voor v2 als de
    grondwaarheid om v1 en v2 tegen te valideren."""
    df = kassa.copy()
    if "datum" not in df.columns:
        df["datum"] = df["tijdstip"].dt.normalize()
    per_dag_dagdeel = df.groupby(["datum", "dagdeel"], observed=True)["regelbedrag"].sum().reset_index()
    dagtotaal = per_dag_dagdeel.groupby("datum")["regelbedrag"].transform("sum")
    per_dag_dagdeel["aandeel"] = per_dag_dagdeel["regelbedrag"] / dagtotaal
    return (
        per_dag_dagdeel.pivot(index="datum", columns="dagdeel", values="aandeel")
        .reindex(columns=DAGDELEN)
        .fillna(0.0)
    )


def dagdeel_aandeel_per_weekday(kassa: pd.DataFrame, train_end: str = TEST_START) -> pd.DataFrame:
    """v1, bewaard als simpele/robuuste referentiebaseline om v2 tegen te
    vergelijken (zie evalueer_dagdeel_aanpak) -- vast historisch aandeel
    per weekdag, reageert niet op weer/events van een specifieke dag."""
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
    return aandeel.div(aandeel.sum(axis=1), axis=0)


def _normaliseer(voorspellingen: dict, index) -> pd.DataFrame:
    """5 losse voorspellingen (kunnen negatief of niet-som-tot-1 zijn) ->
    niet-negatief en genormaliseerd zodat een rij optelt tot 1."""
    arr = np.clip(np.column_stack([voorspellingen[d] for d in DAGDELEN]), 0, None)
    som = arr.sum(axis=1, keepdims=True)
    som[som == 0] = 1.0
    return pd.DataFrame(arr / som, columns=DAGDELEN, index=index)


def train_dagdeel_modellen(daily_features: pd.DataFrame, kassa: pd.DataFrame, test_start: str):
    """v2: traint per dagdeel een HistGradientBoostingRegressor (zelfde
    soort model als model.py) die het aandeel in de dagomzet voorspelt uit
    weekdag/maand/trend/weer/event. daily_features moet dezelfde features
    hebben als het omzetmodel (zie build_features in src/pipeline.py) en
    een 'datum'-kolom om te matchen met de orderregels.

    Retourneert (aandeel_train, aandeel_test) -- allebei een DataFrame met
    kolommen DAGDELEN, per rij genormaliseerd naar 1, in dezelfde volgorde
    als train/test na time_split."""
    aandelen = _dagdeel_aandelen_per_dag(kassa).reset_index()
    df = daily_features.merge(aandelen, on="datum", how="inner")
    train, test = time_split(df, test_start)

    train_pred, test_pred = {}, {}
    for dagdeel in DAGDELEN:
        reg = HistGradientBoostingRegressor(random_state=0, categorical_features=["weekday_num", "month"])
        reg.fit(train[DAGDEEL_FEATURE_COLS], train[dagdeel])
        train_pred[dagdeel] = reg.predict(train[DAGDEEL_FEATURE_COLS])
        test_pred[dagdeel] = reg.predict(test[DAGDEEL_FEATURE_COLS])

    return _normaliseer(train_pred, train.index), _normaliseer(test_pred, test.index)


def evalueer_dagdeel_aanpak(daily_features: pd.DataFrame, kassa: pd.DataFrame, test_start: str) -> pd.DataFrame:
    """Vergelijkt v1 (vast weekdaggemiddelde) met v2 (per-dag model) tegen
    de werkelijke historische dagdeel-verdeling in de testperiode -- dit IS
    te valideren, in tegenstelling tot de rest van het roosteradvies."""
    werkelijk = _dagdeel_aandelen_per_dag(kassa)
    train, test = time_split(daily_features, test_start)

    v1 = dagdeel_aandeel_per_weekday(kassa, test_start)
    v1_per_dag = v1.loc[test["datum"].dt.weekday].to_numpy()
    v1_pred = pd.DataFrame(v1_per_dag, columns=DAGDELEN, index=test.index)

    _, v2_pred = train_dagdeel_modellen(daily_features, kassa, test_start)

    werkelijk_test = werkelijk.reindex(test["datum"]).reset_index(drop=True)
    v1_pred = v1_pred.reset_index(drop=True)
    v2_pred = v2_pred.reset_index(drop=True)

    mae_v1 = (v1_pred - werkelijk_test).abs().mean()
    mae_v2 = (v2_pred - werkelijk_test).abs().mean()
    return pd.DataFrame({"v1_vast_weekdaggemiddelde": mae_v1, "v2_per_dag_model": mae_v2})


def roosteradvies(
    voorspelde_dagomzet: np.ndarray,
    datums: pd.Series,
    aandeel_per_dag,
    norm_omzet_per_uur: float,
    shiftlengte_uur: float = GEMIDDELDE_SHIFTLENGTE_UUR,
) -> pd.DataFrame:
    """Voorspelde dagomzet -> omzet per dagdeel -> uren -> mensen (naar
    boven afgerond). aandeel_per_dag: (n_dagen x len(DAGDELEN)), per dag al
    bepaald aandeel -- van v1 (weekdag-lookup) of v2 (per-dag model), maakt
    deze functie niet uit."""
    aandeel_per_dag = np.asarray(aandeel_per_dag, dtype=float)
    omzet_per_dagdeel = np.asarray(voorspelde_dagomzet, dtype=float)[:, None] * aandeel_per_dag

    uren_per_dagdeel = omzet_per_dagdeel / norm_omzet_per_uur
    mensen_per_dagdeel = np.ceil(uren_per_dagdeel / shiftlengte_uur).astype(int)

    result = pd.DataFrame(mensen_per_dagdeel, columns=DAGDELEN)
    result.insert(0, "datum", np.asarray(datums))
    return result


def main():
    kassa = load_kassa_orderregels()

    df = load_dagstaat()
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    train, test = time_split(df, TEST_START)

    print("validatie: v1 (vast weekdaggemiddelde) vs. v2 (per-dag model), "
          "MAE t.o.v. de werkelijke dagdeel-verdeling (lager is beter):")
    print((evalueer_dagdeel_aanpak(df, kassa, TEST_START) * 100).round(2).to_string())
    print()

    _, aandeel_v2 = train_dagdeel_modellen(df, kassa, TEST_START)
    norm = bepaal_norm_omzet_per_uur(train)
    omzet_pred = model.predict(train, test, model.TARGET)

    advies = roosteradvies(omzet_pred, test["datum"].reset_index(drop=True), aandeel_v2.to_numpy(), norm)
    print("roosteradvies (aantal mensen per dagdeel, v2), eerste 10 dagen van de testperiode:")
    print(advies.head(10).to_string(index=False))
    print()
    print("gemiddeld aantal mensen per dagdeel over de hele testperiode:")
    print(advies[DAGDELEN].mean().round(1).to_string())


if __name__ == "__main__":
    main()
