"""
Weer ophalen voor een locatie, zonder dat de gebruiker een API-key nodig
heeft: Nominatim (OpenStreetMap) voor het opzoeken van een naam/adres naar
coordinaten, Open-Meteo voor het weer zelf.

Twee soorten weer, met opzet niet door elkaar gebruikt:
  - fetch_historical_weather(): wat er ACHTERAF geobserveerd is (archief/
    reanalyse). Prima om ergens op te trainen (het model leert een
    verband, geen voorspelling), maar oneerlijk om een testperiode mee te
    scoren -- op het moment dat je in het echt zou plannen, wist je dit nog
    niet.
  - fetch_forecast_weather(): wat de weersvoorspelling N dagen van tevoren
    daadwerkelijk zei (Open-Meteo's Previous Runs API), dus met realistische
    onzekerheid. Dit is wat je op de dag van plannen ECHT tot je beschikking
    had. Alleen bruikbaar voor een leesperiode vanaf ~begin 2024 (ouder
    archief bestaat niet) en tot 7 dagen vooruit (langer bewaart Open-Meteo
    niet per leadtime).
"""
import requests
import pandas as pd

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
USER_AGENT = "horeca-voorspelling-prototype/1.0"

MAX_LEAD_DAYS = 7  # Open-Meteo's Previous Runs API bewaart niet verder terug per leadtime


def geocode(plaats_of_naam: str) -> dict:
    """Zoekt een naam/adres op naar coordinaten. Werkt ook op namen van
    zaken (bijv. 'Cafe Laurierboom'), niet alleen plaatsnamen."""
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": plaats_of_naam, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"geen locatie gevonden voor '{plaats_of_naam}'")
    r = results[0]
    return {"lat": float(r["lat"]), "lon": float(r["lon"]), "naam": r["display_name"]}


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Dagwaarden temp_c/neerslag_mm/zon_index tussen start_date en end_date
    (inclusief), zoals het WERKELIJK is geweest (archief/reanalyse). zon_index
    is het aandeel van de daglichturen dat er zon was (0-1) -- Open-Meteo
    heeft geen directe 'zonne-index', dit is de dichtstbijzijnde afgeleide."""
    df = _fetch_archive_daily(lat, lon, start_date, end_date)
    return df[["datum", "temp_c", "neerslag_mm", "zon_index"]]


def fetch_forecast_weather(lat: float, lon: float, start_date: str, end_date: str, lead_days: int = MAX_LEAD_DAYS) -> pd.DataFrame:
    """Dagwaarden zoals de voorspelling er `lead_days` dagen van tevoren
    uitzag (Open-Meteo's Previous Runs API) -- voor een eerlijke backtest:
    dit is wat je op het moment van plannen echt had geweten, niet het
    achteraf gemeten weer. Alleen als hourly-data beschikbaar, dus hier
    zelf naar dagwaarden geaggregeerd (temp: gemiddelde, neerslag: som,
    zon: som van zonneschijnduur / daglichtduur -- daglichtduur hangt niet
    af van het weermodel, dus die komt gewoon uit het archief)."""
    suffix = f"previous_day{lead_days}"
    resp = requests.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": f"temperature_2m_{suffix},precipitation_{suffix},sunshine_duration_{suffix}",
            "timezone": "Europe/Amsterdam",
        },
        timeout=60,  # hourly-data over een lange periode duurt langer dan de archief-call
    )
    resp.raise_for_status()
    data = resp.json()["hourly"]

    hourly = pd.DataFrame({
        "tijdstip": pd.to_datetime(data["time"]),
        "temp_c": data[f"temperature_2m_{suffix}"],
        "neerslag_mm": data[f"precipitation_{suffix}"],
        "sunshine_s": data[f"sunshine_duration_{suffix}"],
    })
    hourly["datum"] = hourly["tijdstip"].dt.normalize()
    daily = hourly.groupby("datum", as_index=False).agg(
        temp_c=("temp_c", "mean"),
        neerslag_mm=("neerslag_mm", "sum"),
        sunshine_s=("sunshine_s", "sum"),
    )

    daglicht = _fetch_archive_daily(lat, lon, start_date, end_date)[["datum", "daylight_duration"]]
    daily = daily.merge(daglicht, on="datum", how="left")
    daily["zon_index"] = (daily["sunshine_s"] / daily["daylight_duration"]).clip(0, 1)
    return daily[["datum", "temp_c", "neerslag_mm", "zon_index"]]


def _fetch_archive_daily(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum,sunshine_duration,daylight_duration",
            "timezone": "Europe/Amsterdam",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "datum": pd.to_datetime(data["time"]),
        "temp_c": data["temperature_2m_mean"],
        "neerslag_mm": data["precipitation_sum"],
        "sunshine_duration": data["sunshine_duration"],
        "daylight_duration": data["daylight_duration"],
    })
    df["zon_index"] = (df["sunshine_duration"] / df["daylight_duration"]).clip(0, 1)
    return df
