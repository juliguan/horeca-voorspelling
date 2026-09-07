"""
Streamlit-dashboard bovenop de bestaande pipeline. Roept alleen functies aan
uit model.py, staffing.py, purchasing.py en src/* -- geen van die bestanden
wordt hier aangepast. De enige plek waar dat niet vanzelf werkt is
purchasing.build_calendar(), die intern load_dagstaat() zonder argumenten
aanroept; die naam wordt hieronder na een upload vervangen door een functie
die de geuploade data teruggeeft (module-attribuut overschrijven vanuit dit
bestand, niet de broncode van purchasing.py).

Start: streamlit run app.py
"""
import io

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loading import (
    load_dagstaat as read_dagstaat_csv,
    load_kassa_orderregels as read_kassa_csv,
    load_producten,
    load_verbruik_theoretisch,
    load_inkoop_historie,
)
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split, TEST_START
from src.evaluation import scoreboard
from src.baselines import predict_weekday_average, predict_same_weekday_n_weeks_ago, predict_simple_linear
import model
import staffing
import purchasing
import rooster

ACCENT = "#D97757"   # voorspelling / norm -- wat het model zegt
NEUTRAL = "#8B8680"  # werkelijk -- wat er echt is gebeurd

st.set_page_config(page_title="Horeca drukte- en inkoopvoorspelling", page_icon="🍽️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;650;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    [data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Horeca drukte- en inkoopvoorspelling")
st.caption("Prototype op synthetische kassadata, sep 2023 t/m aug 2026 -- test vanaf maart 2026.")


# ---------------------------------------------------------------------------
# Data uploaden -- zonder upload staat alles op 0/leeg, pas na uploaden
# verschijnen de berekeningen. Bestand weer verwijderen (het kruisje in de
# uploader) zet alles weer terug naar 0.
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
dagstaat_upload = st.sidebar.file_uploader("dagstaat.csv", type="csv")
kassa_upload = st.sidebar.file_uploader("kassa_orderregels.csv", type="csv")
st.sidebar.caption("kassa_orderregels.csv is alleen nodig voor het roosteradvies per dagdeel (sectie 4) -- de andere secties werken puur op dagstaat.csv.")

dagstaat_bytes = dagstaat_upload.getvalue() if dagstaat_upload is not None else None
kassa_bytes = kassa_upload.getvalue() if kassa_upload is not None else None

if dagstaat_upload is not None:
    st.sidebar.caption(f"dagstaat.csv geladen ({dagstaat_upload.size:,} bytes)")
else:
    st.sidebar.caption("nog geen dagstaat.csv geupload")

if kassa_upload is not None:
    st.sidebar.caption(f"kassa_orderregels.csv geladen ({kassa_upload.size:,} bytes)")
else:
    st.sidebar.caption("nog geen kassa_orderregels.csv geupload")


@st.cache_data(show_spinner="dagstaat.csv inlezen...")
def read_dagstaat(data: bytes | None) -> pd.DataFrame | None:
    if data is None:
        return None
    return read_dagstaat_csv(path=io.BytesIO(data))


@st.cache_data(show_spinner="kassa_orderregels.csv inlezen...")
def read_kassa(data: bytes | None) -> pd.DataFrame | None:
    if data is None:
        return None
    return read_kassa_csv(path=io.BytesIO(data))


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
    # selectors is de brede, onzichtbare hit-detectielaag -- de muis hovert
    # feitelijk over DEZE laag (niet over de dunne rule-lijn), dus de tooltip
    # hoort hier, niet op rule.
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


def use_uploaded_dagstaat_for_purchasing(df: pd.DataFrame) -> None:
    """purchasing.build_calendar() roept load_dagstaat() intern aan zonder
    argumenten -- dit vervangt die naam in purchasing.py's eigen namespace
    door de geuploade data, zonder het bestand te wijzigen."""
    purchasing.load_dagstaat = lambda *args, **kwargs: df.copy()


# ---------------------------------------------------------------------------
# Sectie 1 -- omzetvoorspelling (model.py + src/baselines.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="omzetmodel trainen en scoren...")
def compute_omzet(dagstaat_bytes: bytes | None):
    df = read_dagstaat(dagstaat_bytes)
    if df is None:
        return None
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)

    train, test = time_split(df, TEST_START)
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
        "weekdag+maand+weer+event+trend (lineair)": (y_test, pred_linear),
        "zelfde dag 4 weken terug": (y_test, pred_same_weekday),
        "weekdaggemiddelde": (y_test, pred_weekday_avg),
    })
    return chart_df, board


st.header("📈 1. Omzetvoorspelling")
try:
    omzet_result = compute_omzet(dagstaat_bytes)
except Exception as e:
    omzet_result = None
    st.error(f"kon de omzetvoorspelling niet berekenen: {e}")

if omzet_result is None:
    if dagstaat_bytes is None:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("MAPE model", "0%")
            c2.metric("verschil t.o.v. weekdaggemiddelde", "0 pt")
        st.caption("Upload een dagstaat.csv hierboven om de omzetvoorspelling te zien.")
    else:
        st.warning(f"geen testdata gevonden vanaf {TEST_START} in de geuploade dagstaat.csv.")
else:
    chart_df, board = omzet_result
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

    st.caption(f"MAPE-scores op de testperiode (vanaf {TEST_START}):")
    st.dataframe(board.style.format({"mape": "{:.1f}%", "mae": "{:.0f}", "rmse": "{:.0f}"}), width="stretch")

st.divider()


# ---------------------------------------------------------------------------
# Sectie 2 -- personeel (staffing.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="personeelsvertaling berekenen...")
def compute_staffing(dagstaat_bytes: bytes | None):
    raw = read_dagstaat(dagstaat_bytes)
    if raw is None:
        return None, None
    yearly = (
        raw.assign(jaar=raw["datum"].dt.year)
        .groupby("jaar")[["ingeroosterde_uren", "benodigde_uren_norm"]]
        .sum()
    )

    df = add_calendar_features(raw)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    train, test = time_split(df, TEST_START)
    if train.empty or test.empty:
        return yearly, None

    norm = staffing.bepaal_norm_omzet_per_uur(train)
    omzet_pred = model.predict(train, test, model.TARGET)
    uren_pred = staffing.aanbevolen_uren(omzet_pred, norm)

    overbezetting_huidig = staffing.overbezetting_euros(test["ingeroosterde_uren"], test["benodigde_uren_norm"])
    overbezetting_model = staffing.overbezetting_euros(uren_pred, test["benodigde_uren_norm"])

    kpis = {
        "norm": norm,
        "overbezetting_huidig": overbezetting_huidig,
        "overbezetting_model": overbezetting_model,
        "voorkomen": overbezetting_huidig - overbezetting_model,
    }
    return yearly, kpis


st.header("👥 2. Personeel")
try:
    yearly, staffing_kpis = compute_staffing(dagstaat_bytes)
except Exception as e:
    yearly, staffing_kpis = None, None
    st.error(f"kon de personeelsvertaling niet berekenen: {e}")

if yearly is None:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("overbezetting testperiode (huidig rooster)", "€0")
        c2.metric("resterend als je het model volgt", "€0")
        c3.metric("voorkomen door voorspelling", "€0")
    st.caption("Upload een dagstaat.csv hierboven om deze sectie te zien.")
else:
    st.bar_chart(yearly[["ingeroosterde_uren", "benodigde_uren_norm"]], color=[NEUTRAL, ACCENT])
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
        st.warning(f"geen testdata gevonden vanaf {TEST_START} om de personeels-KPI's op te berekenen.")

st.divider()


# ---------------------------------------------------------------------------
# Sectie 3 -- inkoop (purchasing.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="besteladvies per product berekenen (14 modellen)...")
def compute_purchasing(dagstaat_bytes: bytes | None):
    dagstaat_df = read_dagstaat(dagstaat_bytes)
    if dagstaat_df is None:
        return None
    use_uploaded_dagstaat_for_purchasing(dagstaat_df)

    producten = load_producten().set_index("productcode")
    verbruik = load_verbruik_theoretisch()
    inkoop = load_inkoop_historie()
    calendar = purchasing.build_calendar()

    rows = []
    for productcode, product in producten.iterrows():
        result = purchasing.run_product(productcode, product, verbruik, calendar, inkoop)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows) if rows else None


st.header("📦 3. Inkoop")
try:
    inkoop_board = compute_purchasing(dagstaat_bytes)
except Exception as e:
    inkoop_board = None
    st.error(f"kon het besteladvies niet berekenen: {e}")

if inkoop_board is None:
    if dagstaat_bytes is None:
        with st.container(border=True):
            st.metric("totaal verschil (advies - werkelijk)", "€0")
        st.caption("Upload een dagstaat.csv hierboven om het besteladvies te zien.")
    else:
        st.warning(f"geen testdata gevonden vanaf {TEST_START} om het besteladvies op te berekenen.")
else:
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

st.divider()


# ---------------------------------------------------------------------------
# Sectie 4 -- roosteradvies per dagdeel (rooster.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="roosteradvies per dagdeel berekenen...")
def compute_rooster(dagstaat_bytes: bytes | None, kassa_bytes: bytes | None):
    kassa = read_kassa(kassa_bytes)
    if kassa is None:
        return None

    df = read_dagstaat(dagstaat_bytes)
    if df is None:
        return None
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)

    train, test = time_split(df, TEST_START)
    if train.empty or test.empty:
        return None

    aandeel = rooster.dagdeel_aandeel_per_weekday(kassa, TEST_START)
    norm = staffing.bepaal_norm_omzet_per_uur(train)
    omzet_pred = model.predict(train, test, model.TARGET)
    return rooster.roosteradvies(omzet_pred, test["datum"].reset_index(drop=True), aandeel, norm)


st.header("🧑‍🍳 4. Roosteradvies per dagdeel")
st.caption(
    "Vereist zowel dagstaat.csv als kassa_orderregels.csv. Puur een advies, geen "
    "vergelijking met werkelijk -- dagstaat.csv heeft alleen een dagtotaal aan "
    "ingeroosterde uren, geen uitsplitsing per dagdeel om tegenaan te leggen."
)
try:
    rooster_advies = compute_rooster(dagstaat_bytes, kassa_bytes)
except Exception as e:
    rooster_advies = None
    st.error(f"kon het roosteradvies niet berekenen: {e}")

if rooster_advies is None:
    with st.container(border=True):
        st.bar_chart(pd.Series(0, index=rooster.DAGDELEN, name="gemiddeld aantal mensen"), color=ACCENT)
    if dagstaat_bytes is None or kassa_bytes is None:
        st.caption("Upload zowel dagstaat.csv als kassa_orderregels.csv hierboven om deze sectie te zien.")
    else:
        st.warning(f"geen testdata gevonden vanaf {TEST_START}.")
else:
    gemiddeld = rooster_advies[rooster.DAGDELEN].mean()
    gemiddeld.name = "gemiddeld aantal mensen"
    with st.container(border=True):
        st.bar_chart(gemiddeld, color=ACCENT)
    st.caption(f"roosteradvies per dag, testperiode (vanaf {TEST_START}):")
    st.dataframe(rooster_advies, width="stretch")
