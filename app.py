"""
Streamlit-dashboard bovenop de bestaande pipeline. Roept alleen functies aan
uit model.py, staffing.py, purchasing.py, rooster.py en src/* -- geen van
die bestanden wordt hier aangepast.

Hoofdinvoer is nu 1 bestand: je eigen orderregels (zoals kassa_orderregels.csv).
Daaruit wordt alles afgeleid:
  - dagomzet (src/aggregation.aggregate_orders_to_daily) -- voor het omzetmodel
  - productverbruik via de receptuur (src/aggregation.aggregate_orders_to_verbruik)
    -- voor inkoop, dezelfde vertaalslag als verbruik_theoretisch.csv origineel had
  - dagdeel-aandelen (rooster.dagdeel_aandeel_per_weekday) -- voor het rooster
Het weer wordt er zelf bij gezocht (src/weather.py, Nominatim + Open-Meteo,
geen API-key nodig) op basis van een locatie die je opgeeft.

dagstaat.csv en inkoop_historie.csv zijn optioneel: alleen nodig om te
vergelijken met wat je nu al doet (sectie 2 en 3). Zonder die bestanden
krijg je nog steeds het volledige advies, alleen geen "werkelijk"-kolom.

purchasing.build_calendar() roept intern load_dagstaat() aan zonder
argumenten -- die naam wordt hieronder vervangen door een functie die de
zelf opgebouwde dag+weer-tabel teruggeeft (module-attribuut overschrijven
vanuit dit bestand, niet de broncode van purchasing.py).

Start: streamlit run app.py
"""
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data_loading import (
    load_dagstaat as read_dagstaat_csv,
    load_kassa_orderregels as read_orders_csv,
    load_inkoop_historie as read_inkoop_csv,
    load_producten,
    load_receptuur,
)
from src.aggregation import aggregate_orders_to_daily, aggregate_orders_to_verbruik
from src.weather import geocode, fetch_historical_weather
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split
from src.evaluation import scoreboard
from src.baselines import predict_weekday_average, predict_same_weekday_n_weeks_ago, predict_simple_linear
import model
import staffing
import purchasing
import rooster

ACCENT = "#D97757"   # voorspelling / advies -- wat het model zegt
NEUTRAL = "#8B8680"  # werkelijk -- wat er echt is gebeurd

LOGO_PATH = Path(__file__).parent / "logo_icon.png"

st.set_page_config(page_title="Drukmeter", page_icon=Image.open(LOGO_PATH), layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;650;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    [data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }

    @keyframes drukmeter-needle-sweep {
        0%   { transform: rotate(-70deg); }
        60%  { transform: rotate(8deg); }
        100% { transform: rotate(0deg); }
    }
    @keyframes drukmeter-fade-in {
        0%   { opacity: 0; transform: translateY(6px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .drukmeter-header { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
    .drukmeter-needle {
        transform-origin: 40px 50px;
        animation: drukmeter-needle-sweep 0.9s cubic-bezier(.2,.8,.2,1) both;
    }
    .drukmeter-wordmark {
        font-size: 2.6rem; font-weight: 650; letter-spacing: -0.02em; color: #EDEAE5;
        animation: drukmeter-fade-in 0.6s ease-out 0.5s both;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="drukmeter-header">
      <svg width="56" height="56" viewBox="0 0 80 100">
        <circle cx="40" cy="50" r="32" fill="none" stroke="#D97757" stroke-width="5"/>
        <circle cx="40" cy="50" r="5" fill="#D97757"/>
        <line class="drukmeter-needle" x1="40" y1="50" x2="60" y2="30" stroke="#D97757" stroke-width="5" stroke-linecap="round"/>
      </svg>
      <span class="drukmeter-wordmark">Drukmeter</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Upload je eigen orderregels -- de rest (weer, drukte, inkoop, rooster) volgt daaruit.")


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


# ---------------------------------------------------------------------------
# Data uploaden -- zonder bestellingen staat alles op 0/leeg.
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
orders_upload = st.sidebar.file_uploader("je orderregels (zoals kassa_orderregels.csv)", type="csv")
locatie = st.sidebar.text_input("locatie (voor het weer)", value="Cafe Laurierboom, Amsterdam")

if locatie:
    try:
        _gevonden = cached_geocode(locatie)
        render_location_badge(st.sidebar, gevonden_naam=_gevonden["naam"])
    except Exception:
        render_location_badge(st.sidebar)

norm_omzet_per_uur = st.sidebar.number_input("omzet-norm per gewerkt uur (€)", min_value=1.0, value=100.0, step=1.0)

st.sidebar.divider()
st.sidebar.caption("optioneel -- alleen voor vergelijking met wat je nu al doet:")
dagstaat_upload = st.sidebar.file_uploader("dagstaat.csv (huidig rooster)", type="csv")
inkoop_upload = st.sidebar.file_uploader("inkoop_historie.csv (huidige bestellingen)", type="csv")

orders_bytes = orders_upload.getvalue() if orders_upload is not None else None
dagstaat_bytes = dagstaat_upload.getvalue() if dagstaat_upload is not None else None
inkoop_bytes = inkoop_upload.getvalue() if inkoop_upload is not None else None

if orders_upload is not None:
    st.sidebar.caption(f"orderregels geladen ({orders_upload.size:,} bytes)")
else:
    st.sidebar.caption("nog geen orderregels geupload")


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
    """Orderregels -> dagomzet, aangevuld met opgehaald weer. Dit is de
    tabel die zowel het omzetmodel als het rooster (via de kalender/weer-
    kolommen) verder gebruiken."""
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


def build_omzet_chart(chart_df: pd.DataFrame) -> alt.LayerChart:
    """st.line_chart snapt bij veel dagpunten op een smalle grafiek al bij een
    piepklein muisbewegingetje meerdere dagen door (elke dag is maar een paar
    pixels breed). Dit is dezelfde data, maar met een expliciete 'dichtstbijzijnde
    dag'-selectie (het standaard Altair hover-patroon) zodat de tooltip per
    dag vastklikt."""
    wide = chart_df.reset_index()
    reeksen = [c for c in wide.columns if c != "datum"]
    long_df = wide.melt("datum", var_name="reeks", value_name="omzet")

    kleuren = alt.Scale(domain=reeksen, range=[NEUTRAL, ACCENT])
    nearest = alt.selection_point(nearest=True, on="pointermove", fields=["datum"], empty=False)

    lines = alt.Chart(long_df).mark_line().encode(
        x=alt.X("datum:T", title=None),
        y=alt.Y("omzet:Q", title="omzet (€)"),
        color=alt.Color("reeks:N", scale=kleuren, title=None, legend=alt.Legend(orient="bottom")),
    )
    selectors = alt.Chart(wide).mark_point().encode(
        x="datum:T",
        opacity=alt.value(0),
        tooltip=[alt.Tooltip("datum", type="temporal", title="datum", format="%a %d %b %Y")]
        + [alt.Tooltip(r, type="quantitative", title=r, format=",.0f") for r in reeksen],
    ).add_params(nearest)
    points = lines.mark_point(size=45).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    rule = alt.Chart(wide).mark_rule(color=NEUTRAL).encode(
        x="datum:T",
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
    ).transform_filter(nearest)

    return alt.layer(lines, selectors, points, rule).properties(height=380)


def build_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(base_df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    return df


# ---------------------------------------------------------------------------
# Sectie 1 -- omzetvoorspelling (model.py + src/baselines.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="omzetmodel trainen en scoren...")
def compute_omzet(orders_bytes: bytes | None, locatie: str):
    base = build_base_df(orders_bytes, locatie)
    if base is None:
        return None
    df = build_features(base)
    train, test, test_start = maak_test_split(df)
    if train.empty or test.empty:
        return None

    y_test = test[model.TARGET]
    pred_gbm = model.predict(train, test, model.TARGET)
    pred_weekday_avg = predict_weekday_average(train, test, model.TARGET)
    pred_same_weekday = predict_same_weekday_n_weeks_ago(df, test["datum"], model.TARGET)
    pred_linear = predict_simple_linear(train, test, model.TARGET)

    chart_df = pd.DataFrame(
        {"werkelijk": y_test.to_numpy(), "voorspelling (model)": pred_gbm},
        index=pd.DatetimeIndex(test["datum"], name="datum"),
    )
    board = scoreboard({
        "model (HistGradientBoostingRegressor)": (y_test, pred_gbm),
        "weekdag+maand+weer+trend (lineair)": (y_test, pred_linear),
        "zelfde dag 4 weken terug": (y_test, pred_same_weekday),
        "weekdaggemiddelde": (y_test, pred_weekday_avg),
    })
    return chart_df, board, test_start


st.header("📈 1. Omzetvoorspelling")
try:
    omzet_result = compute_omzet(orders_bytes, locatie)
except Exception as e:
    omzet_result = None
    st.error(f"kon de omzetvoorspelling niet berekenen: {e}")

if omzet_result is None:
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("MAPE model", "0%")
        c2.metric("verschil t.o.v. weekdaggemiddelde", "0 pt")
    if orders_bytes is None:
        st.caption("Upload je orderregels hierboven om de omzetvoorspelling te zien.")
    else:
        st.warning("te weinig dagen in de upload om een testperiode van te maken.")
else:
    chart_df, board, test_start = omzet_result
    model_mape = board.loc[board.model == "model (HistGradientBoostingRegressor)", "mape"].iloc[0]
    baseline_mape = board.loc[board.model == "weekdaggemiddelde", "mape"].iloc[0]

    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("MAPE model", f"{model_mape:.1f}%")
        c2.metric(
            "verschil t.o.v. weekdaggemiddelde",
            f"{model_mape - baseline_mape:+.1f} pt",
            delta=f"{model_mape - baseline_mape:+.1f} pt",
            delta_color="inverse",
        )
        st.altair_chart(build_omzet_chart(chart_df), width="stretch")

    st.caption(f"MAPE-scores op de testperiode (vanaf {test_start}, laatste 20% van je data):")
    st.dataframe(board.style.format({"mape": "{:.1f}%", "mae": "{:.0f}", "rmse": "{:.0f}"}), width="stretch")

st.divider()


# ---------------------------------------------------------------------------
# Sectie 2 -- personeel (staffing.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="personeelsadvies berekenen...")
def compute_staffing(orders_bytes: bytes | None, locatie: str, norm: float, dagstaat_bytes: bytes | None):
    base = build_base_df(orders_bytes, locatie)
    if base is None:
        return None
    df = build_features(base)
    train, test, test_start = maak_test_split(df)
    if train.empty or test.empty:
        return None

    omzet_pred = model.predict(train, test, model.TARGET)
    uren_pred = staffing.aanbevolen_uren(omzet_pred, norm)
    advies = pd.DataFrame({"datum": test["datum"].to_numpy(), "aanbevolen_uren": uren_pred})

    dagstaat = read_dagstaat(dagstaat_bytes)
    if dagstaat is None:
        return advies, None

    vergelijk = advies.merge(
        dagstaat[["datum", "ingeroosterde_uren", "benodigde_uren_norm"]], on="datum", how="left"
    )
    overbezetting_huidig = staffing.overbezetting_euros(vergelijk["ingeroosterde_uren"], vergelijk["benodigde_uren_norm"])
    overbezetting_model = staffing.overbezetting_euros(vergelijk["aanbevolen_uren"], vergelijk["benodigde_uren_norm"])
    kpis = {
        "overbezetting_huidig": overbezetting_huidig,
        "overbezetting_model": overbezetting_model,
        "voorkomen": overbezetting_huidig - overbezetting_model,
    }
    return advies, kpis


st.header("👥 2. Personeel")
try:
    staffing_result = compute_staffing(orders_bytes, locatie, norm_omzet_per_uur, dagstaat_bytes)
except Exception as e:
    staffing_result = None
    st.error(f"kon het personeelsadvies niet berekenen: {e}")

if staffing_result is None:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("overbezetting testperiode (huidig rooster)", "€0")
        c2.metric("resterend als je het model volgt", "€0")
        c3.metric("voorkomen door voorspelling", "€0")
    if orders_bytes is None:
        st.caption("Upload je orderregels hierboven om deze sectie te zien.")
    else:
        st.warning("te weinig dagen in de upload om een testperiode van te maken.")
else:
    advies, staffing_kpis = staffing_result
    chart_data = advies.set_index("datum")["aanbevolen_uren"].rename("aanbevolen uren")
    st.bar_chart(chart_data, color=ACCENT)

    if staffing_kpis is not None:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("overbezetting testperiode (huidig rooster)", f"€{staffing_kpis['overbezetting_huidig']:,.0f}")
            c2.metric("resterend als je het model volgt", f"€{staffing_kpis['overbezetting_model']:,.0f}")
            c3.metric(
                "voorkomen door voorspelling",
                f"€{staffing_kpis['voorkomen']:,.0f}",
                delta=f"€{staffing_kpis['voorkomen']:,.0f}",
            )
    else:
        st.caption("Upload ook dagstaat.csv (optioneel, links) om te zien hoe dit zich verhoudt tot je huidige rooster.")

st.divider()


# ---------------------------------------------------------------------------
# Sectie 3 -- inkoop (purchasing.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="besteladvies per product berekenen (14 modellen)...")
def compute_purchasing(orders_bytes: bytes | None, locatie: str, inkoop_bytes: bytes | None):
    orders = read_orders(orders_bytes)
    base = build_base_df(orders_bytes, locatie)
    if orders is None or base is None:
        return None
    _, _, test_start = maak_test_split(base)
    use_base_df_for_purchasing(base)
    # purchasing.run_product() gebruikt intern TEST_START uit src/split.py
    # voor de train/test-split per product -- die staat vast op de
    # synthetische testperiode, dus die zetten we hier op de datum die past
    # bij de geuploade data.
    purchasing.TEST_START = test_start

    receptuur = load_receptuur()
    verbruik = aggregate_orders_to_verbruik(orders, receptuur)

    inkoop = read_inkoop(inkoop_bytes)
    heeft_inkoop_historie = inkoop is not None
    if inkoop is None:
        inkoop = pd.DataFrame({
            "leverdatum": pd.Series(dtype="datetime64[ns]"),
            "productcode": pd.Series(dtype="str"),
            "aantal": pd.Series(dtype="float"),
            "bedrag": pd.Series(dtype="float"),
        })

    producten = load_producten().set_index("productcode")
    calendar = purchasing.build_calendar()

    rows = []
    for productcode, product in producten.iterrows():
        result = purchasing.run_product(productcode, product, verbruik, calendar, inkoop)
        if result is not None:
            rows.append(result)
    if not rows:
        return None
    return pd.DataFrame(rows), heeft_inkoop_historie


st.header("📦 3. Inkoop")
try:
    purchasing_result = compute_purchasing(orders_bytes, locatie, inkoop_bytes)
except Exception as e:
    purchasing_result = None
    st.error(f"kon het besteladvies niet berekenen: {e}")

if purchasing_result is None:
    with st.container(border=True):
        st.metric("totaal besteladvies", "€0")
    if orders_bytes is None:
        st.caption("Upload je orderregels hierboven om het besteladvies te zien.")
    else:
        st.warning("te weinig dagen in de upload om een testperiode van te maken.")
else:
    inkoop_board, heeft_inkoop_historie = purchasing_result

    if heeft_inkoop_historie:
        show_cols = [
            "productcode", "product", "besteleenheid",
            "werkelijk_eenheden", "besteladvies_eenheden", "verschil_eenheden",
            "werkelijk_euro", "besteladvies_euro", "verschil_euro",
        ]
        totaal = {
            "productcode": "TOTAAL", "product": "", "besteleenheid": "",
            "werkelijk_eenheden": inkoop_board.werkelijk_eenheden.sum(),
            "besteladvies_eenheden": inkoop_board.besteladvies_eenheden.sum(),
            "verschil_eenheden": inkoop_board.verschil_eenheden.sum(),
            "werkelijk_euro": inkoop_board.werkelijk_euro.sum(),
            "besteladvies_euro": inkoop_board.besteladvies_euro.sum(),
            "verschil_euro": inkoop_board.verschil_euro.sum(),
        }
        display_board = pd.concat([inkoop_board[show_cols], pd.DataFrame([totaal])], ignore_index=True)
        st.dataframe(
            display_board.style.format({
                "werkelijk_eenheden": "{:,.0f}", "besteladvies_eenheden": "{:,.0f}", "verschil_eenheden": "{:+,.0f}",
                "werkelijk_euro": "€{:,.0f}", "besteladvies_euro": "€{:,.0f}", "verschil_euro": "€{:+,.0f}",
            }),
            width="stretch",
        )
        with st.container(border=True):
            st.metric(
                "totaal verschil (advies - werkelijk)",
                f"€{totaal['verschil_euro']:+,.0f}",
                delta=f"€{totaal['verschil_euro']:+,.0f}",
            )
    else:
        show_cols = ["productcode", "product", "besteleenheid", "besteladvies_eenheden", "besteladvies_euro"]
        totaal_euro = inkoop_board.besteladvies_euro.sum()
        st.dataframe(
            inkoop_board[show_cols].style.format({"besteladvies_eenheden": "{:,.0f}", "besteladvies_euro": "€{:,.0f}"}),
            width="stretch",
        )
        with st.container(border=True):
            st.metric("totaal besteladvies (testperiode)", f"€{totaal_euro:,.0f}")
        st.caption("Upload ook inkoop_historie.csv (optioneel, links) om te zien hoe dit zich verhoudt tot wat je nu bestelt.")

st.divider()


# ---------------------------------------------------------------------------
# Sectie 4 -- roosteradvies per dagdeel (rooster.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="roosteradvies per dagdeel berekenen...")
def compute_rooster(orders_bytes: bytes | None, locatie: str, norm: float):
    orders = read_orders(orders_bytes)
    base = build_base_df(orders_bytes, locatie)
    if orders is None or base is None:
        return None
    df = build_features(base)
    train, test, test_start = maak_test_split(df)
    if train.empty or test.empty:
        return None

    aandeel = rooster.dagdeel_aandeel_per_weekday(orders, test_start)
    omzet_pred = model.predict(train, test, model.TARGET)
    return rooster.roosteradvies(omzet_pred, test["datum"].reset_index(drop=True), aandeel, norm)


st.header("🧑‍🍳 4. Roosteradvies per dagdeel")
st.caption("Puur een advies, geen vergelijking met werkelijk -- er is geen 'werkelijk aantal mensen per dagdeel' om tegenaan te leggen.")
try:
    rooster_advies = compute_rooster(orders_bytes, locatie, norm_omzet_per_uur)
except Exception as e:
    rooster_advies = None
    st.error(f"kon het roosteradvies niet berekenen: {e}")

if rooster_advies is None:
    with st.container(border=True):
        st.bar_chart(pd.Series(0, index=rooster.DAGDELEN, name="gemiddeld aantal mensen"), color=ACCENT)
    if orders_bytes is None:
        st.caption("Upload je orderregels hierboven om deze sectie te zien.")
    else:
        st.warning("te weinig dagen in de upload om een testperiode van te maken.")
else:
    gemiddeld = rooster_advies[rooster.DAGDELEN].mean()
    gemiddeld.name = "gemiddeld aantal mensen"
    with st.container(border=True):
        st.bar_chart(gemiddeld, color=ACCENT)
    st.caption("roosteradvies per dag, testperiode:")
    st.dataframe(rooster_advies, width="stretch")
