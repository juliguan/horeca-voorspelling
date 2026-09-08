"""
Gedeelde look-and-feel voor nieuwe pagina's (niet voor app.py zelf, die
behoudt zijn eigen inline versie om 'm onaangeroerd te laten). Zelfde
lettertype, kleuren en micro-interacties als de hoofdpagina, zonder het
opstartscherm -- dat hoort bij de allereerste keer laden van de app, niet
bij het wisselen tussen pagina's.
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
