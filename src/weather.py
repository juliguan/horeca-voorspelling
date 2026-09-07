"""
Weer ophalen voor een locatie, zonder dat de gebruiker een API-key nodig
heeft: Nominatim (OpenStreetMap) voor het opzoeken van een naam/adres naar
coordinaten, Open-Meteo voor de historische dagwaarden.
"""
import requests
import pandas as pd

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "horeca-voorspelling-prototype/1.0"


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
    (inclusief), voor de gegeven coordinaten. zon_index is het aandeel van
    de daglichturen dat er zon was (0-1) -- Open-Meteo heeft geen directe
    'zonne-index', dit is de dichtstbijzijnde afgeleide."""
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
        "zon_index": pd.Series(data["sunshine_duration"]) / pd.Series(data["daylight_duration"]),
    })
    df["zon_index"] = df["zon_index"].clip(0, 1)
    return df
