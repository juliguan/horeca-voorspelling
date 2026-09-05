"""
Besteladvies per inkoopproduct: voorspel het verbruik tot de volgende
leverdag en zet dat om in hele besteleenheden, met een houdbaarheidsplafond.
Los van model.py/staffing.py -- geen wijzigingen daar, dit is een eigen
pipeline per productcode.

Aanpak per product:
  1. dagelijkse reeks theoretisch_verbruik (verbruik_theoretisch.csv), aangevuld
     met dezelfde soort features als het omzetmodel: kalender, weer, event-vlag,
     lags/rolling van het eigen verbruik (allemaal shift-based, geen lekkage).
  2. leverdagen (producten.csv) bepalen de beslismomenten: op elke leverdag
     moet het verbruik tot de VOLGENDE leverdag voorspeld worden -- dat is het
     target, en "aantal dagen tot de volgende levering" is er zelf een feature
     bij (verschilt per product en per leverdag-combinatie).
  3. HistGradientBoostingRegressor per product, getraind op log1p(target)
     (kan net als bij het omzetmodel geen 0 aan, en sommige producten hebben
     wel eens een verbruik van (bijna) 0 in een venster).
  4. besteladvies = voorspeld verbruik, naar boven afgerond op hele
     besteleenheden, geplafonneerd op wat binnen de houdbaarheid nog op kan
     (geschat met het recente daggemiddelde verbruik van dat product).

Run: source .venv/bin/activate && python purchasing.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_loading import (
    load_dagstaat,
    load_producten,
    load_verbruik_theoretisch,
    load_inkoop_historie,
    load_derving_werkelijk,
)
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split, TEST_START

VERBRUIK_COL = "theoretisch_verbruik"

DAG_MAP = {"ma": 0, "di": 1, "wo": 2, "do": 3, "vr": 4, "za": 5, "zo": 6}

FEATURE_COLS = [
    "weekday_num", "month", "trend", "day_of_year",
    "temp_c", "neerslag_mm", "zon_index", "is_droog",
    "is_event",
    "dagen_tot_volgende_levering",
    f"{VERBRUIK_COL}_lag_7", f"{VERBRUIK_COL}_lag_14",
    f"{VERBRUIK_COL}_lag_21", f"{VERBRUIK_COL}_lag_28",
    f"{VERBRUIK_COL}_rollmean_7", f"{VERBRUIK_COL}_rollstd_7",
    f"{VERBRUIK_COL}_rollmean_28", f"{VERBRUIK_COL}_rollstd_28",
]


def parse_leverdagen(s: str) -> list:
    return sorted(DAG_MAP[d.strip()] for d in s.split(","))


def dagen_tot_volgende_levering(weekday_today: int, leverdagen: list) -> int:
    """Kleinste k>=1 zodat (weekday_today + k) % 7 een leverdag is."""
    for k in range(1, 8):
        if (weekday_today + k) % 7 in leverdagen:
            return k
    raise ValueError("geen leverdag gevonden binnen 7 dagen")


def build_calendar() -> pd.DataFrame:
    """Kalender/weer/event-features op dagniveau, zoals bij het omzetmodel."""
    cal = load_dagstaat()[["datum", "temp_c", "neerslag_mm", "zon_index", "event"]]
    cal = add_calendar_features(cal)
    cal = add_event_flags(cal)
    cal = add_weather_features(cal)
    return cal[["datum", "weekday_num", "month", "trend", "day_of_year", "temp_c", "neerslag_mm", "zon_index", "is_droog", "is_event"]]


def build_product_daily(verbruik: pd.DataFrame, productcode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    """Volledige dagelijkse reeks voor 1 product: verbruik (0 aangevuld op
    ontbrekende dagen -- die betekenen 0 verkocht, niet onbekend) + features."""
    sub = verbruik[verbruik.productcode == productcode][["datum", VERBRUIK_COL]]
    daily = calendar.merge(sub, on="datum", how="left")
    daily[VERBRUIK_COL] = daily[VERBRUIK_COL].fillna(0.0)
    daily = add_lag_features(daily, target=VERBRUIK_COL)
    daily = add_rolling_features(daily, target=VERBRUIK_COL)
    return daily


def build_decision_points(daily: pd.DataFrame, leverdagen: list) -> pd.DataFrame:
    """1 rij per leverdag, met als target het verbruik tot de volgende
    leverdag (som van de tussenliggende dagen, cumsum-truc, geen lekkage
    want alleen bekende/toekomstige dagen binnen hetzelfde venster)."""
    daily = daily.reset_index(drop=True)
    n = len(daily)
    gap_by_weekday = {wd: dagen_tot_volgende_levering(wd, leverdagen) for wd in range(7)}
    horizon = daily["weekday_num"].map(gap_by_weekday).to_numpy()

    verbruik = daily[VERBRUIK_COL].to_numpy()
    cumsum = np.concatenate([[0.0], np.cumsum(verbruik)])
    idx = np.arange(n)
    valid_window = (idx + horizon) <= n
    target = np.full(n, np.nan)
    target[valid_window] = cumsum[idx[valid_window] + horizon[valid_window]] - cumsum[idx[valid_window]]

    decision = daily.copy()
    decision["dagen_tot_volgende_levering"] = horizon
    decision["target_verbruik_tot_volgende_levering"] = target

    is_delivery_day = decision["weekday_num"].isin(leverdagen)
    decision = decision[is_delivery_day & decision["target_verbruik_tot_volgende_levering"].notna()]
    return decision.reset_index(drop=True)


def predict_verbruik(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """HistGradientBoostingRegressor op log1p(target), standaardparameters."""
    X_train = train[FEATURE_COLS]
    y_train = np.log1p(train["target_verbruik_tot_volgende_levering"])
    X_test = test[FEATURE_COLS]

    reg = HistGradientBoostingRegressor(random_state=0, categorical_features=["weekday_num", "month"])
    reg.fit(X_train, y_train)
    return np.expm1(reg.predict(X_test))


def besteladvies(voorspeld_verbruik: np.ndarray, test: pd.DataFrame, product: pd.Series) -> pd.DataFrame:
    """Voorspeld verbruik -> hele besteleenheden, geplafonneerd op wat binnen
    de houdbaarheid nog verbruikt kan worden (recente daggemiddelde x
    houdbaarheid_dagen als schatting van wat 'opkan')."""
    daggemiddelde = test[f"{VERBRUIK_COL}_rollmean_28"].fillna(test[f"{VERBRUIK_COL}_rollmean_7"])
    max_binnen_houdbaarheid = daggemiddelde.to_numpy() * product["houdbaarheid_dagen"]

    order_basis = np.minimum(voorspeld_verbruik, max_binnen_houdbaarheid)
    order_basis = np.maximum(order_basis, 0.0)

    eenheden = np.ceil(order_basis / product["inhoud_per_besteleenheid"]).astype(int)
    euro = eenheden * product["inkoopprijs_per_besteleenheid"]

    return pd.DataFrame({
        "datum": test["datum"].to_numpy(),
        "voorspeld_verbruik": voorspeld_verbruik,
        "max_binnen_houdbaarheid": max_binnen_houdbaarheid,
        "besteladvies_eenheden": eenheden,
        "besteladvies_euro": euro,
    })


def werkelijk_besteld(inkoop: pd.DataFrame, productcode: str) -> pd.DataFrame:
    sub = inkoop[inkoop.productcode == productcode]
    return sub.groupby("leverdatum", as_index=False).agg(
        werkelijk_eenheden=("aantal", "sum"),
        werkelijk_euro=("bedrag", "sum"),
    ).rename(columns={"leverdatum": "datum"})


def run_product(productcode: str, product: pd.Series, verbruik: pd.DataFrame, calendar: pd.DataFrame, inkoop: pd.DataFrame) -> dict:
    leverdagen = parse_leverdagen(product["leverdagen"])
    daily = build_product_daily(verbruik, productcode, calendar)
    decision = build_decision_points(daily, leverdagen)

    train, test = time_split(decision, TEST_START)
    if len(test) == 0:
        return None

    voorspeld = predict_verbruik(train, test)
    advies = besteladvies(voorspeld, test, product)

    # left-merge op advies (bevat alleen testperiode-leverdagen), zodat orders
    # van vóór de testperiode niet meetellen -- anders telt de vergelijking
    # de hele historie uit inkoop_historie mee i.p.v. alleen de testperiode.
    werkelijk = werkelijk_besteld(inkoop, productcode)
    vergelijk = advies.merge(werkelijk, on="datum", how="left").fillna(
        {"werkelijk_eenheden": 0, "werkelijk_euro": 0.0}
    )

    return {
        "productcode": productcode,
        "product": product["product"],
        "besteleenheid": product["besteleenheid"],
        "n_leverdagen_test": len(vergelijk),
        "werkelijk_eenheden": vergelijk["werkelijk_eenheden"].sum(),
        "besteladvies_eenheden": vergelijk["besteladvies_eenheden"].sum(),
        "werkelijk_euro": vergelijk["werkelijk_euro"].sum(),
        "besteladvies_euro": vergelijk["besteladvies_euro"].sum(),
        "verschil_eenheden": vergelijk["besteladvies_eenheden"].sum() - vergelijk["werkelijk_eenheden"].sum(),
        "verschil_euro": vergelijk["besteladvies_euro"].sum() - vergelijk["werkelijk_euro"].sum(),
    }


def derving_check(resultaten: pd.DataFrame, productcode_to_row: dict) -> None:
    """Achteraf-check, NIET gebruikt om op te trainen: had het advies minder
    ingekocht dan er werkelijk is bedorven suggereert, of niet?"""
    derving = load_derving_werkelijk()
    derving_test = derving[derving.datum >= TEST_START]
    if derving_test.empty:
        print("geen bederf-data in de testperiode.")
        return

    print("achteraf-check tegen derving_werkelijk.csv (alleen ter info, niet gebruikt om te trainen):")
    for productcode, sub in derving_test.groupby("productcode"):
        bedorven_euro = sub["kostprijs"].sum()
        bedorven_eenheden_basis = sub["aantal_bedorven"].sum()
        row = resultaten[resultaten.productcode == productcode]
        if row.empty:
            continue
        row = row.iloc[0]
        inhoud = productcode_to_row[productcode]["inhoud_per_besteleenheid"]
        verschil_basiseenheid = (row["besteladvies_eenheden"] - row["werkelijk_eenheden"]) * inhoud
        richting = "minder" if verschil_basiseenheid < 0 else "meer of evenveel"
        print(f"  {productcode}: werkelijk bedorven €{bedorven_euro:,.2f} ({bedorven_eenheden_basis:.0f} {sub.basiseenheid.iloc[0]}) "
              f"-- advies had over de testperiode {richting} ingekocht ({verschil_basiseenheid:+.1f} {sub.basiseenheid.iloc[0]} t.o.v. werkelijk)")


def main():
    producten = load_producten().set_index("productcode")
    verbruik = load_verbruik_theoretisch()
    inkoop = load_inkoop_historie()
    calendar = build_calendar()

    rows = []
    for productcode, product in producten.iterrows():
        result = run_product(productcode, product, verbruik, calendar, inkoop)
        if result is not None:
            rows.append(result)

    board = pd.DataFrame(rows)
    print(f"scoreboard per product -- testperiode vanaf {TEST_START}, werkelijk besteld vs. besteladvies:")
    print(board[[
        "productcode", "product", "besteleenheid",
        "werkelijk_eenheden", "besteladvies_eenheden", "verschil_eenheden",
        "werkelijk_euro", "besteladvies_euro", "verschil_euro",
    ]].to_string(index=False))

    print()
    print(f"totaal werkelijk besteld:   €{board.werkelijk_euro.sum():,.2f}")
    print(f"totaal besteladvies:        €{board.besteladvies_euro.sum():,.2f}")
    print(f"totaal verschil:            €{board.verschil_euro.sum():,.2f}")
    print()

    derving_check(board, producten.to_dict("index"))


if __name__ == "__main__":
    main()
