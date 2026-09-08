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
import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.data_loading import load_producten, load_receptuur
from src.aggregation import aggregate_orders_to_verbruik
from src.evaluation import scoreboard
from src.baselines import predict_weekday_average, predict_same_weekday_n_weeks_ago, predict_simple_linear
from src.pipeline import (
    ACCENT, NEUTRAL, EVENTS_PATH, LOGO_PATH,
    load_local_events, cached_geocode, render_location_badge,
    read_orders, read_dagstaat, read_inkoop, cached_weather,
    build_base_df, maak_test_split, use_base_df_for_purchasing, build_features,
)
import model
import staffing
import purchasing
import rooster
import stockout_detectie

st.set_page_config(page_title="Vooruitzicht", page_icon=Image.open(LOGO_PATH), layout="wide")

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

    /* Uber-achtige micro-interacties: snappy, springy easing i.p.v. lineair,
       duidelijke druk-feedback, content die rustig infadet i.p.v. abrupt verschijnt. */
    button, [data-testid="stFileUploaderDropzone"], [data-baseweb="input"], [data-baseweb="base-input"] {
        transition: transform 0.15s cubic-bezier(.34,1.56,.64,1), border-color 0.2s ease, box-shadow 0.2s ease;
    }
    button:active { transform: scale(0.96); }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: #D97757 !important; }
    [data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {
        box-shadow: 0 0 0 2px rgba(217,119,87,0.35);
    }
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] {
        animation: drukmeter-fade-in 0.5s cubic-bezier(.2,.8,.2,1) both;
    }
    [data-testid="stAltairChart"], [data-testid="stArrowVegaLiteChart"] {
        animation: drukmeter-fade-in 0.7s cubic-bezier(.2,.8,.2,1) both;
    }

    /* Opstartscherm, Uber-stijl: effen donker vlak, gecentreerd woordmerk,
       geen spinner -- kort in beeld, dan wegfaden zodat de dashboard eronder
       zichtbaar wordt. pointer-events:none zodat 'ie nooit iets blokkeert. */
    /* De achtergrond zelf is METEEN (t=0) volledig ondoorzichtig en blijft
       dat tot vlak voor het einde -- geen fade-in van het zwarte vlak zelf,
       want dan schemert de sidebar er in dat eerste fractie-van-een-seconde
       nog doorheen. Alleen het logo/woord erin krijgt een zachte intro. */
    @keyframes vooruitzicht-splash-bg {
        0%   { opacity: 1; }
        85%  { opacity: 1; }
        100% { opacity: 0; }
    }
    @keyframes vooruitzicht-splash-content-in {
        0%   { opacity: 0; transform: scale(0.92); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes vooruitzicht-splash-needle {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .vooruitzicht-splash {
        position: fixed; inset: 0; z-index: 9999; pointer-events: none;
        background: #141313;
        display: flex; align-items: center; justify-content: center;
        animation: vooruitzicht-splash-bg 4.5s linear forwards;
    }
    .vooruitzicht-splash-content {
        display: flex; flex-direction: column; align-items: center; gap: 22px;
        animation: vooruitzicht-splash-content-in 0.6s cubic-bezier(.2,.8,.2,1) both;
    }
    .vooruitzicht-splash-word {
        font-size: 3.4rem; font-weight: 650; letter-spacing: -0.02em; color: #EDEAE5;
    }
    .vooruitzicht-splash-needle {
        transform-origin: 60px 75px;
        animation: vooruitzicht-splash-needle 1.1s linear infinite;
    }
    /* Streamlit rendert de sidebar in een eigen laag die niet onder de
       fixed overlay hierboven valt -- apart afdekken met een ::before op
       de sidebar zelf, zelfde achtergrond-animatie zodat het synchroon oogt. */
    [data-testid="stSidebar"] { position: relative; }
    [data-testid="stSidebar"]::before {
        content: ""; position: absolute; inset: 0; z-index: 9999; pointer-events: none;
        background: #141313;
        animation: vooruitzicht-splash-bg 4.5s linear forwards;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="vooruitzicht-splash">
      <div class="vooruitzicht-splash-content">
        <svg width="88" height="88" viewBox="0 0 120 150">
          <circle cx="60" cy="75" r="48" fill="none" stroke="#D97757" stroke-width="7"/>
          <circle cx="60" cy="75" r="7" fill="#D97757"/>
          <line class="vooruitzicht-splash-needle" x1="60" y1="75" x2="90" y2="45" stroke="#D97757" stroke-width="7" stroke-linecap="round"/>
        </svg>
        <span class="vooruitzicht-splash-word">Vooruitzicht</span>
      </div>
    </div>
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
      <span class="drukmeter-wordmark">Vooruitzicht</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Upload je eigen orderregels -- de rest (weer, drukte, inkoop, rooster) volgt daaruit.")

_events_df = load_local_events()
if _events_df is not None:
    _aankomend = _events_df[_events_df["datum"] >= pd.Timestamp.now().normalize()].sort_values("datum")
    if not _aankomend.empty:
        _afstand_kleur = {"dichtbij": ACCENT, "stad-breed": NEUTRAL, "ver weg": "#6B6560"}
        with st.expander(f"📍 {len(_aankomend)} lokale evenementen gevonden (automatisch onderzocht)", expanded=False):
            st.caption("Wordt gebruikt als event-signaal in de omzetvoorspelling -- elke 1e/15e automatisch bijgewerkt.")
            for _, _ev in _aankomend.iterrows():
                _kleur = _afstand_kleur.get(_ev["afstand_schatting"], NEUTRAL)
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:12px; padding:8px 0; '
                    f'border-bottom:1px solid #302D2B;">'
                    f'<span style="font-variant-numeric:tabular-nums; color:#8B8680; min-width:90px;">'
                    f'{_ev["datum"].strftime("%d %b %Y")}</span>'
                    f'<span style="flex:1;">{_ev["naam"]}</span>'
                    f'<span style="font-size:0.75rem; color:{_kleur}; border:1px solid {_kleur}; '
                    f'border-radius:999px; padding:2px 10px;">{_ev["afstand_schatting"]}</span>'
                    f'</div>',
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
    aandeel_per_dag = aandeel.loc[test["datum"].dt.weekday].to_numpy()
    omzet_pred = model.predict(train, test, model.TARGET)
    return rooster.roosteradvies(omzet_pred, test["datum"].reset_index(drop=True), aandeel_per_dag, norm)


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

st.divider()


# ---------------------------------------------------------------------------
# Sectie 5 -- vermoedelijke nee-verkopen (stockout_detectie.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="vermoedelijke nee-verkopen zoeken...")
def compute_stockouts(orders_bytes: bytes | None):
    orders = read_orders(orders_bytes)
    if orders is None:
        return None
    receptuur = load_receptuur()
    producten = load_producten().set_index("productcode")

    verdacht = stockout_detectie.detecteer_vermoedelijke_stockouts(orders, receptuur)
    if verdacht.empty:
        return verdacht, None

    verdacht = verdacht.merge(producten[["product"]], left_on="productcode", right_index=True)
    samenvatting = (
        verdacht.groupby(["productcode", "product"])
        .agg(aantal_keer=("datum", "count"), totaal_misgelopen=("geschatte_misgelopen_omzet", "sum"))
        .reset_index()
        .sort_values("totaal_misgelopen", ascending=False)
    )
    return verdacht, samenvatting


st.header("🚫 5. Vermoedelijke nee-verkopen")
st.caption(
    "Heuristiek, geen zekerheid: dagen waarop een product opvallend vroeg stopte met verkopen "
    "terwijl de zaak die dag verder gemiddeld-tot-druk was -- een teken dat het waarschijnlijk "
    "vroegtijdig op was. Werkt puur op de orderregels zelf, geen aparte 'uitverkocht'-registratie nodig."
)
try:
    stockout_result = compute_stockouts(orders_bytes)
except Exception as e:
    stockout_result = None
    st.error(f"kon de nee-verkopen niet berekenen: {e}")

if stockout_result is None:
    with st.container(border=True):
        st.metric("geschatte misgelopen omzet", "€0")
    if orders_bytes is None:
        st.caption("Upload je orderregels hierboven om deze sectie te zien.")
else:
    verdacht, samenvatting = stockout_result
    if samenvatting is None or samenvatting.empty:
        with st.container(border=True):
            st.metric("geschatte misgelopen omzet", "€0")
        st.caption("Geen vermoedelijke nee-verkopen gevonden in deze data.")
    else:
        totaal = samenvatting["totaal_misgelopen"].sum()
        with st.container(border=True):
            st.metric("geschatte misgelopen omzet (hele periode)", f"€{totaal:,.0f}")
        st.caption(f"per product, {len(verdacht)} gevlagde momenten in totaal:")
        st.dataframe(
            samenvatting.rename(columns={
                "aantal_keer": "keer gevlagd", "totaal_misgelopen": "geschatte misgelopen omzet",
            }).style.format({"geschatte misgelopen omzet": "€{:,.0f}"}),
            width="stretch",
        )
