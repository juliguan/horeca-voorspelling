"""
Eigen pagina, puur gericht op het roosteradvies -- los van de bredere
dashboard (die blijft ongewijzigd op de hoofdpagina staan). Zelfde
onderliggende pijplijn (src/pipeline.py, model.py, rooster.py), maar hier
expliciet in beeld: het advies combineert 3 signalen -- je eigen
orderregels, het weer, en lokale evenementen -- niet alleen drukte-uit-het-
verleden zoals een simpel rooster op basis van "vorige week".

Eigen upload hier (los van de hoofdpagina): dit is een aparte Streamlit-
pagina, dus een eigen bestand kiezen hoort erbij.

Start: streamlit run app.py  (deze pagina verschijnt dan in de navigatie)
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.branding import inject_css, render_header
from src.pipeline import (
    ACCENT, NEUTRAL, LOGO_PATH,
    load_local_events, cached_geocode, render_location_badge,
    read_orders, build_base_df, maak_test_split, build_features,
)
import model
import rooster

st.set_page_config(page_title="Rooster · Vooruitzicht", page_icon=Image.open(LOGO_PATH), layout="wide")
inject_css()
render_header("Vooruitzicht — Rooster")
st.caption("Eén pagina, puur gericht op personeelsplanning: je orders, het weer en lokale evenementen samen.")

# ---------------------------------------------------------------------------
# Eigen, kleine sidebar -- bewust los van de hoofdpagina
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
orders_bytes = orders_upload.getvalue() if orders_upload is not None else None


@st.cache_data(show_spinner="roosteradvies + onderliggende signalen berekenen...")
def compute_rooster_pagina(orders_bytes: bytes | None, locatie: str, norm: float):
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
    advies = rooster.roosteradvies(omzet_pred, test["datum"].reset_index(drop=True), aandeel, norm)

    weer_samenvatting = {
        "gem_temp": test["temp_c"].mean(),
        "regendagen": int((test["neerslag_mm"] > 1.0).sum()),
        "totaal_dagen": len(test),
    }
    return advies, test["datum"], weer_samenvatting


if orders_bytes is None:
    st.info("Upload je orderregels in de zijbalk om het roosteradvies te zien.")
else:
    try:
        resultaat = compute_rooster_pagina(orders_bytes, locatie, norm_omzet_per_uur)
    except Exception as e:
        resultaat = None
        st.error(f"kon het roosteradvies niet berekenen: {e}")

    if resultaat is None:
        st.warning("te weinig dagen in de upload om een testperiode van te maken.")
    else:
        advies, test_datums, weer = resultaat

        # -- de 3 signalen zichtbaar maken, niet alleen gebruiken --
        st.subheader("Op basis van 3 signalen samen")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("**🧾 Je orders**")
                st.caption(f"{test_datums.nunique()} dagen in de testperiode, per dagdeel uitgesplitst")
        with c2:
            with st.container(border=True):
                st.markdown("**🌤️ Het weer**")
                st.caption(f"gem. {weer['gem_temp']:.1f}°C, {weer['regendagen']} regenachtige dagen van de {weer['totaal_dagen']}")
        with c3:
            with st.container(border=True):
                events = load_local_events()
                if events is not None:
                    in_periode = events[(events["datum"] >= test_datums.min()) & (events["datum"] <= test_datums.max())]
                else:
                    in_periode = pd.DataFrame()
                st.markdown("**📍 Lokale evenementen**")
                st.caption(f"{in_periode['naam'].nunique() if not in_periode.empty else 0} gevonden binnen de testperiode")

        if not in_periode.empty:
            with st.expander(f"welke evenementen vallen in deze periode ({in_periode['naam'].nunique()})", expanded=False):
                for naam, groep in in_periode.groupby("naam"):
                    dagen = groep["datum"].dt.strftime("%d %b").tolist()
                    st.markdown(f"- **{naam}** — {', '.join(dagen)} ({groep['afstand_schatting'].iloc[0]})")

        st.divider()

        # -- het advies zelf --
        st.subheader("Roosteradvies per dagdeel")
        gemiddeld = advies[rooster.DAGDELEN].mean()
        gemiddeld.name = "gemiddeld aantal mensen"
        with st.container(border=True):
            st.bar_chart(gemiddeld, color=ACCENT)

        totaal_mensen_gemiddeld = gemiddeld.sum()
        st.metric("gemiddeld totaal aantal mensen per dag (alle dagdelen samen)", f"{totaal_mensen_gemiddeld:.1f}")

        st.caption("roosteradvies per dag, hele testperiode:")
        st.dataframe(advies, width="stretch")
