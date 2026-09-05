"""
Vertaalslag van voorspelde omzet (model.py) naar aanbevolen personeelsuren en
euro's. Geen nieuwe features, geen aanpassing aan het omzetmodel zelf --
alleen omzet -> uren -> euro's, om te zien hoeveel van de bestaande
overbezetting een voorspelling had kunnen voorkomen op de testperiode.

Run: source .venv/bin/activate && python staffing.py
"""
import numpy as np
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
from model import predict as predict_omzet, TARGET

UURLOON = 16.80


def bepaal_norm_omzet_per_uur(train: pd.DataFrame) -> float:
    """Norm-omzet per gewerkt uur, herberekend uit de trainperiode als
    omzet / benodigde_uren_norm (die kolom staat al voor elke dag in
    dagstaat.csv, dit middelt 'm over train tot 1 constante)."""
    dagen = train[train["benodigde_uren_norm"] > 0]
    norm_per_dag = dagen["omzet"] / dagen["benodigde_uren_norm"]
    return float(norm_per_dag.mean())


def aanbevolen_uren(voorspelde_omzet, norm_omzet_per_uur: float) -> np.ndarray:
    """Voorspelde omzet omgerekend naar aanbevolen uren, met dezelfde norm
    die het bedrijf al gebruikt voor benodigde_uren_norm."""
    return np.asarray(voorspelde_omzet, dtype=float) / norm_omzet_per_uur


def overbezetting_euros(ingeroosterd, benodigd, uurloon: float = UURLOON) -> float:
    """Kosten van overbezetting: alleen de dagen waarop er meer uren op het
    rooster stonden dan benodigd, in euro's. (Dit is de definitie die ook de
    eerder gemeten ~€96.000 over 3 jaar oplevert -- zie sanity check hieronder.)"""
    verschil_uren = np.asarray(ingeroosterd, dtype=float) - np.asarray(benodigd, dtype=float)
    return float(np.clip(verschil_uren, a_min=0, a_max=None).sum() * uurloon)


def verschil_euros(a, b, uurloon: float = UURLOON) -> float:
    """Netto verschil tussen twee uren-reeksen, in euro's (kan positief of
    negatief zijn -- positief betekent a duurder/meer uren dan b)."""
    return float((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum() * uurloon)


def main():
    df = load_dagstaat()
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=TARGET)
    df = add_rolling_features(df, target=TARGET)

    train, test = time_split(df, TEST_START)

    norm = bepaal_norm_omzet_per_uur(train)
    print(f"norm-omzet per gewerkt uur (uit trainperiode): €{norm:.2f}")

    # sanity check: overbezetting over de volle 3 jaar, puur uit de data,
    # geen model nodig -- dit hoort in de buurt van de eerder gemeten ~€96.000 te komen
    overbezetting_3jaar = overbezetting_euros(df["ingeroosterde_uren"], df["benodigde_uren_norm"])
    print(f"sanity check -- overbezetting hele periode (3 jaar) uit de data: €{overbezetting_3jaar:,.0f} "
          f"(eerder gemeten: ~€96.000)")
    print()

    # huidige overbezetting op de testperiode (rooster vs. de norm die achteraf bleek te kloppen)
    overbezetting_test_huidig = overbezetting_euros(test["ingeroosterde_uren"], test["benodigde_uren_norm"])

    # model-voorspelde omzet -> aanbevolen uren, alleen op de testperiode
    omzet_pred = predict_omzet(train, test, TARGET)
    uren_pred = aanbevolen_uren(omzet_pred, norm)

    # netto verschil tussen wat er echt is ingeroosterd en wat het model had aanbevolen
    verschil_huidig_vs_model = verschil_euros(test["ingeroosterde_uren"], uren_pred)

    # overbezetting die zou overblijven als je het rooster op de model-aanbeveling had gebaseerd
    overbezetting_test_model = overbezetting_euros(uren_pred, test["benodigde_uren_norm"])

    voorkomen = overbezetting_test_huidig - overbezetting_test_model

    print("scoreboard -- overbezetting in euro's, testperiode "
          f"({test['datum'].min().date()} t/m {test['datum'].max().date()}):")
    print(f"  huidige overbezetting (rooster vs. norm):                    €{overbezetting_test_huidig:>9,.0f}")
    print(f"  verschil rooster vs. model-aanbeveling (netto, +/-):         €{verschil_huidig_vs_model:>9,.0f}")
    print(f"  resterende overbezetting als je het model had gevolgd:       €{overbezetting_test_model:>9,.0f}")
    print(f"  -> door een voorspelling voorkomen op de testperiode:        €{voorkomen:>9,.0f}")
    print(f"     (dat is {voorkomen / overbezetting_test_huidig * 100:.0f}% van de overbezetting "
          f"die er op de testperiode al zat)")


if __name__ == "__main__":
    main()
