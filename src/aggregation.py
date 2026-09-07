"""
Ruwe orderregels (zoals kassa_orderregels.csv) omzetten naar de twee vormen
die de rest van de pipeline nodig heeft: een dagomzet-reeks (voor het
omzetmodel/rooster) en een productverbruik-reeks (voor inkoop), via de
receptuur -- dezelfde vertaalslag als waarmee verbruik_theoretisch.csv
oorspronkelijk is gemaakt.
"""
import pandas as pd


def aggregate_orders_to_daily(orders: pd.DataFrame) -> pd.DataFrame:
    """orders met kolommen tijdstip/datum en regelbedrag/order_id -> 1 rij
    per dag met omzet en aantal_orders. Vult ontbrekende dagen (geen enkele
    order die dag) aan met 0 -- dat is een geldige waarde, geen onbekende."""
    df = orders.copy()
    if "datum" not in df.columns:
        df["datum"] = df["tijdstip"].dt.normalize()

    daily = df.groupby("datum").agg(
        omzet=("regelbedrag", "sum"),
        aantal_orders=("order_id", "nunique"),
    )
    volledige_reeks = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(volledige_reeks, fill_value=0)
    daily.index.name = "datum"
    return daily.reset_index()


def aggregate_orders_to_verbruik(orders: pd.DataFrame, receptuur: pd.DataFrame) -> pd.DataFrame:
    """orders (met item/aantal) + receptuur (item -> productcode, verbruik_per_stuk)
    -> theoretisch verbruik per product per dag. Zelfde vorm als
    verbruik_theoretisch.csv (datum, productcode, theoretisch_verbruik)."""
    df = orders.copy()
    if "datum" not in df.columns:
        df["datum"] = df["tijdstip"].dt.normalize()

    merged = df.merge(receptuur, on="item", how="inner")
    merged["verbruik"] = merged["aantal"] * merged["verbruik_per_stuk"]

    verbruik = (
        merged.groupby(["datum", "productcode"])["verbruik"]
        .sum()
        .reset_index()
        .rename(columns={"verbruik": "theoretisch_verbruik"})
    )
    return verbruik
