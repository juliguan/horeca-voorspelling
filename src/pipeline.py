"""
Gedeelde databouwstenen voor de Streamlit-pagina's (app.py en pages/*.py).
Puur data-opbouw/caching, geen sectie-rendering -- dat blijft in de
paginabestanden zelf, zodat elke pagina zijn eigen lay-out/verhaal houdt
terwijl de onderliggende pijplijn (orders -> dagomzet -> weer -> events ->
features) overal precies hetzelfde werkt.

Verplaatst uit app.py zonder gedragswijziging (zelfde functies, zelfde
caching) zodat de nieuwe rooster-pagina dit kan hergebruiken in plaats van
te dupliceren.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loading import (
    load_dagstaat as read_dagstaat_csv,
    load_kassa_orderregels as read_orders_csv,
    load_inkoop_historie as read_inkoop_csv,
)
from src.aggregation import aggregate_orders_to_daily
from src.weather import geocode, fetch_historical_weather
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split
import model
import purchasing

ACCENT = "#D97757"   # voorspelling / advies -- wat het model zegt
NEUTRAL = "#8B8680"  # werkelijk -- wat er echt is gebeurd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = PROJECT_ROOT / "data" / "events_handmatig.csv"
LOGO_PATH = PROJECT_ROOT / "logo_icon.png"


def load_local_events() -> pd.DataFrame | None:
    """Lokale evenementen, aangevuld door een terugkerende cloud-taak (elke
    1e/15e van de maand een agent die research doet en dit bestand bijwerkt
    via git). Bestaat het bestand nog niet, dan is er simpelweg nog geen
    event-signaal -- geen fout.

    Het bestand wordt geschreven door een geautomatiseerde taak zonder
    menselijke controle vooraf -- een parsefout hierin mag de rest van de
    app niet meetrekken, dus dit faalt zacht met een waarschuwing."""
    if not EVENTS_PATH.exists():
        return None
    try:
        ev = pd.read_csv(EVENTS_PATH, parse_dates=["datum"])
    except Exception as e:
        st.sidebar.warning(f"kon events_handmatig.csv niet lezen: {e}")
        return None
    return ev if not ev.empty else None


@st.cache_data(show_spinner="locatie opzoeken...")
def cached_geocode(plaats: str) -> dict:
    return geocode(plaats)


def render_location_badge(container, gevonden_naam: str | None = None) -> None:
    """Klein, opvallend chipje i.p.v. een tekstregel die onderaan wegvalt."""
    if gevonden_naam is not None:
        kort = gevonden_naam.split(",")[0]
        kleur, achtergrond, icoon, tekst = "#4CAF50", "rgba(76,175,80,0.15)", "✓", kort
    else:
        kleur, achtergrond, icoon, tekst = ACCENT, "rgba(217,119,87,0.15)", "⚠", "locatie niet gevonden"

    container.markdown(
        f'<div style="display:inline-block; margin-top:-8px; margin-bottom:8px; '
        f'padding:5px 12px; border-radius:999px; background:{achtergrond}; '
        f'border:1px solid {kleur}; color:{kleur}; font-size:0.8rem; font-weight:600;">'
        f'{icoon}&nbsp;&nbsp;{tekst}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="orderregels inlezen...")
def read_orders(data: bytes | None) -> pd.DataFrame | None:
    if data is None:
        return None
    return read_orders_csv(path=io.BytesIO(data))


@st.cache_data(show_spinner="dagstaat.csv inlezen...")
def read_dagstaat(data: bytes | None) -> pd.DataFrame | None:
    if data is None:
        return None
    return read_dagstaat_csv(path=io.BytesIO(data))


@st.cache_data(show_spinner="inkoop_historie.csv inlezen...")
def read_inkoop(data: bytes | None) -> pd.DataFrame | None:
    if data is None:
        return None
    return read_inkoop_csv(path=io.BytesIO(data))


@st.cache_data(show_spinner="weerdata ophalen...")
def cached_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    return fetch_historical_weather(lat, lon, start, end)


@st.cache_data(show_spinner="dagomzet en weer opbouwen...")
def build_base_df(orders_bytes: bytes | None, locatie: str) -> pd.DataFrame | None:
    """Orderregels -> dagomzet, aangevuld met opgehaald weer en lokale
    evenementen. Dit is de tabel die zowel het omzetmodel als het rooster
    (via de kalender/weer/event-kolommen) verder gebruiken."""
    orders = read_orders(orders_bytes)
    if orders is None:
        return None
    daily = aggregate_orders_to_daily(orders)

    loc = cached_geocode(locatie)
    start = daily["datum"].min().strftime("%Y-%m-%d")
    end = daily["datum"].max().strftime("%Y-%m-%d")
    weather = cached_weather(loc["lat"], loc["lon"], start, end)

    df = daily.merge(weather, on="datum", how="left")
    # Open-Meteo's archief heeft een paar dagen vertraging -- de laatste
    # dagen missen dan nog. Vul dat met de laatst bekende waarde in plaats
    # van die dagen helemaal te laten vallen.
    df[["temp_c", "neerslag_mm", "zon_index"]] = df[["temp_c", "neerslag_mm", "zon_index"]].ffill()

    events = load_local_events()
    if events is not None:
        naam_per_datum = events.drop_duplicates("datum").set_index("datum")["naam"]
        df["event"] = df["datum"].map(naam_per_datum)
    else:
        df["event"] = None
    return df


def maak_test_split(df: pd.DataFrame, test_fractie: float = 0.2):
    """Laatste test_fractie van de dagen als testperiode -- vast op
    2026-03-01 zoals bij de synthetische data kan hier niet, want een
    eigen upload heeft een eigen datumbereik."""
    dates = np.sort(df["datum"].unique())
    idx = int(len(dates) * (1 - test_fractie))
    test_start = pd.Timestamp(dates[idx]).strftime("%Y-%m-%d")
    train, test = time_split(df, test_start)
    return train, test, test_start


def use_base_df_for_purchasing(base_df: pd.DataFrame) -> None:
    """purchasing.build_calendar() roept load_dagstaat() intern aan zonder
    argumenten -- dit vervangt die naam in purchasing.py's eigen namespace
    door de zelf opgebouwde dag+weer-tabel, zonder het bestand te wijzigen."""
    cols = base_df[["datum", "temp_c", "neerslag_mm", "zon_index", "event"]]
    purchasing.load_dagstaat = lambda *args, **kwargs: cols.copy()


def build_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(base_df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    return df
