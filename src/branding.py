"""
Gedeelde look-and-feel voor nieuwe pagina's (niet voor app.py zelf, die
behoudt zijn eigen inline versie om 'm onaangeroerd te laten). Zelfde
lettertype, kleuren en micro-interacties als de hoofdpagina, zonder het
opstartscherm -- dat hoort bij de allereerste keer laden van de app, niet
bij het wisselen tussen pagina's.

LET OP: alle st.markdown-strings hieronder staan met opzet volledig
links uitgelijnd (geen Python-inspringing in de string zelf), ook al
staan ze in een ingesprongen functie. Markdown interpreteert 4+ spaties
inspringing als een codeblok -- met de normale Python-inspringing (8
spaties, want de string zit in een functie) werd de HTML als platte
tekst getoond in plaats van gerenderd. Dit is de robuuste oplossing,
niet cosmetisch.
"""
import streamlit as st

from src.pipeline import ACCENT


def inject_css() -> None:
    st.markdown(
"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;650;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
[data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }

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
@keyframes drukmeter-fade-in {
    0%   { opacity: 0; transform: translateY(6px); }
    100% { opacity: 1; transform: translateY(0); }
}
.drukmeter-header { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
.drukmeter-wordmark {
    font-size: 2.6rem; font-weight: 650; letter-spacing: -0.02em; color: #EDEAE5;
    animation: drukmeter-fade-in 0.6s ease-out 0.3s both;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_page_transition(nieuw_label: str, icon_svg_paths: str) -> None:
    """Kort, eigen 'aankomst'-schermpje voor secundaire pagina's: het
    meter-icoontje wisselt (cross-fade) naar een pagina-specifiek icoontje.
    Streamlit heeft geen echte cross-page animatie (elke pagina is een eigen
    scriptrun, geen gedeelde DOM-tijdlijn) -- dit simuleert het gevoel van
    transformatie in het korte moment na aankomst op de nieuwe pagina, in
    plaats van een letterlijke overgang tussen twee pagina's.

    icon_svg_paths: de SVG-inhoud (paths/shapes) van het nieuwe icoon, in
    hetzelfde 120x150 coordinatenstelsel als het meter-icoon."""
    st.markdown(
f"""
<style>
@keyframes rooster-splash-bg {{ 0% {{opacity:1}} 80% {{opacity:1}} 100% {{opacity:0}} }}
@keyframes rooster-icon-out {{
    0% {{ opacity:1; transform:scale(1) rotate(0deg); }}
    100% {{ opacity:0; transform:scale(0.7) rotate(12deg); }}
}}
@keyframes rooster-icon-in {{
    0% {{ opacity:0; transform:scale(0.7) rotate(-12deg); }}
    100% {{ opacity:1; transform:scale(1) rotate(0deg); }}
}}
@keyframes rooster-word-fade {{ 0% {{opacity:0}} 100% {{opacity:1}} }}
.rooster-splash {{
    position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    background: #141313;
    display: flex; align-items: center; justify-content: center;
    animation: rooster-splash-bg 3s linear forwards;
}}
.rooster-splash-inner {{ display:flex; flex-direction:column; align-items:center; gap:22px; }}
.rooster-splash-icons {{ position: relative; width: 88px; height: 88px; }}
.rooster-splash-icons svg {{ position: absolute; top: 0; left: 0; }}
.rooster-icon-gauge {{ animation: rooster-icon-out 0.5s cubic-bezier(.2,.8,.2,1) 0.25s both; }}
.rooster-icon-new {{ animation: rooster-icon-in 0.5s cubic-bezier(.2,.8,.2,1) 0.65s both; }}
.rooster-splash-word {{
    font-size: 2.2rem; font-weight: 650; letter-spacing: -0.02em; color: #EDEAE5;
    animation: rooster-word-fade 0.5s ease 0.85s both;
}}
[data-testid="stSidebar"] {{ position: relative; }}
[data-testid="stSidebar"]::before {{
    content: ""; position: absolute; inset: 0; z-index: 9999; pointer-events: none;
    background: #141313;
    animation: rooster-splash-bg 3s linear forwards;
}}
</style>
<div class="rooster-splash">
<div class="rooster-splash-inner">
<div class="rooster-splash-icons">
<svg class="rooster-icon-gauge" width="88" height="88" viewBox="0 0 120 150">
<circle cx="60" cy="75" r="48" fill="none" stroke="{ACCENT}" stroke-width="7"/>
<circle cx="60" cy="75" r="7" fill="{ACCENT}"/>
<line x1="60" y1="75" x2="90" y2="45" stroke="{ACCENT}" stroke-width="7" stroke-linecap="round"/>
</svg>
<svg class="rooster-icon-new" width="88" height="88" viewBox="0 0 120 150">
{icon_svg_paths}
</svg>
</div>
<span class="rooster-splash-word">Vooruitzicht — {nieuw_label}</span>
</div>
</div>
""",
        unsafe_allow_html=True,
    )


ROOSTER_ICON_SVG = f"""<rect x="30" y="45" width="60" height="55" rx="8" fill="none" stroke="{ACCENT}" stroke-width="6"/>
<line x1="30" y1="65" x2="90" y2="65" stroke="{ACCENT}" stroke-width="6"/>
<line x1="48" y1="35" x2="48" y2="52" stroke="{ACCENT}" stroke-width="6" stroke-linecap="round"/>
<line x1="72" y1="35" x2="72" y2="52" stroke="{ACCENT}" stroke-width="6" stroke-linecap="round"/>"""


def render_header(titel: str = "Vooruitzicht") -> None:
    st.markdown(
f"""
<div class="drukmeter-header">
<svg width="56" height="56" viewBox="0 0 80 100">
<circle cx="40" cy="50" r="32" fill="none" stroke="{ACCENT}" stroke-width="5"/>
<circle cx="40" cy="50" r="5" fill="{ACCENT}"/>
<line x1="40" y1="50" x2="60" y2="30" stroke="{ACCENT}" stroke-width="5" stroke-linecap="round"/>
</svg>
<span class="drukmeter-wordmark">{titel}</span>
</div>
""",
        unsafe_allow_html=True,
    )
