"""
Detecteert waarschijnlijke nee-verkopen (stockouts): dagen waarop een
product vermoedelijk eerder op was dan de vraag stopte, puur af te leiden
uit de orderregels zelf -- geen aparte "uitverkocht"-registratie nodig.
Los bestand, hergebruikt src/aggregation.py, verandert niks aan model.py/
staffing.py/purchasing.py/rooster.py.

Aanpak: een product waar geen vraag naar was, verkoopt de hele avond een
beetje door. Een product dat op is, stopt abrupt en vroeg -- terwijl de
zaak verder nog gewoon druk is. Dat verschil is het signaal:

  1. Laatste verkooptijdstip per product per dag.
  2. Normale laatste-verkooptijd voor dat product op die weekdag (mediaan
     over de geschiedenis).
  3. Vlag een dag als de laatste verkoop significant vroeger is dan normaal
     EN de zaak die dag gemiddeld-tot-druk was (dus niet gewoon een rustige
     avond waarop toevallig niks laat verkocht werd).
  4. Schat de misgelopen omzet: het gemiddelde omzettempo van dat product in
     het gemiste tijdvak, op vergelijkbare (niet-verdachte) dagen van
     dezelfde weekdag.

LET OP: dit is een heuristiek, geen zekerheid -- een vroege laatste verkoop
kan ook toeval zijn. Het is bedoeld als aanwijzing om na te lopen, niet als
harde waarheid. Op de synthetische testdata simuleert de datagenerator geen
echte stockouts, dus resultaten daarop laten zien dat de code werkt, niet
dat de detectie "klopt" -- dat is pas te beoordelen op echte data.

Run: source .venv/bin/activate && python stockout_detectie.py
"""
import numpy as np
import pandas as pd

from src.data_loading import load_kassa_orderregels, load_receptuur, load_producten
from src.aggregation import aggregate_orders_to_daily

DRUKTE_ONDERGRENS = 0.9   # dag moet minstens dit aandeel van het weekdaggemiddelde omzetten
MIN_UUR_VROEGER = 1.5     # laatste verkoop moet minstens dit veel eerder zijn dan normaal


def _met_productcode(orders: pd.DataFrame, receptuur: pd.DataFrame) -> pd.DataFrame:
    df = orders.merge(receptuur[["item", "productcode"]], on="item", how="inner")
    df["datum"] = df["tijdstip"].dt.normalize()
    df["uur_van_dag"] = df["tijdstip"].dt.hour + df["tijdstip"].dt.minute / 60
    df["weekday_num"] = df["datum"].dt.weekday
    return df


def detecteer_vermoedelijke_stockouts(orders: pd.DataFrame, receptuur: pd.DataFrame) -> pd.DataFrame:
    df = _met_productcode(orders, receptuur)

    laatste = (
        df.groupby(["datum", "productcode", "weekday_num"])["uur_van_dag"]
        .max()
        .rename("laatste_verkoop_uur")
        .reset_index()
    )
    normaal = (
        laatste.groupby(["productcode", "weekday_num"])["laatste_verkoop_uur"]
        .median()
        .rename("normale_laatste_verkoop_uur")
        .reset_index()
    )
    laatste = laatste.merge(normaal, on=["productcode", "weekday_num"])
    laatste["uren_vroeger"] = laatste["normale_laatste_verkoop_uur"] - laatste["laatste_verkoop_uur"]

    daily = aggregate_orders_to_daily(orders)
    daily["weekday_num"] = daily["datum"].dt.weekday
    daily["drukte_aandeel"] = daily["omzet"] / daily.groupby("weekday_num")["omzet"].transform("mean")
    laatste = laatste.merge(daily[["datum", "drukte_aandeel"]], on="datum")

    verdacht = laatste[
        (laatste["uren_vroeger"] >= MIN_UUR_VROEGER) & (laatste["drukte_aandeel"] >= DRUKTE_ONDERGRENS)
    ].copy()

    schattingen = []
    for _, row in verdacht.iterrows():
        vergelijkbaar = df[
            (df["productcode"] == row["productcode"])
            & (df["weekday_num"] == row["weekday_num"])
            & (df["uur_van_dag"] >= row["laatste_verkoop_uur"])
            & (df["uur_van_dag"] < row["normale_laatste_verkoop_uur"])
            & (df["datum"] != row["datum"])
        ]
        n_dagen = vergelijkbaar["datum"].nunique()
        schattingen.append(vergelijkbaar["regelbedrag"].sum() / n_dagen if n_dagen else 0.0)

    verdacht["geschatte_misgelopen_omzet"] = schattingen
    return verdacht.sort_values("datum").reset_index(drop=True)


def main():
    orders = load_kassa_orderregels()
    receptuur = load_receptuur()
    producten = load_producten().set_index("productcode")

    verdacht = detecteer_vermoedelijke_stockouts(orders, receptuur)
    verdacht = verdacht.merge(producten[["product"]], left_on="productcode", right_index=True)

    print(f"{len(verdacht)} vermoedelijke stockout-momenten gevonden "
          f"(drempel: >={MIN_UUR_VROEGER}u vroeger dan normaal, op een dag met "
          f">={DRUKTE_ONDERGRENS*100:.0f}% van het weekdaggemiddelde)")
    print()
    cols = ["datum", "productcode", "product", "laatste_verkoop_uur", "normale_laatste_verkoop_uur",
            "uren_vroeger", "drukte_aandeel", "geschatte_misgelopen_omzet"]
    print(verdacht[cols].to_string(index=False))
    print()
    print(f"totaal geschat misgelopen omzet: €{verdacht['geschatte_misgelopen_omzet'].sum():,.2f}")


if __name__ == "__main__":
    main()
