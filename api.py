"""
Backend-API voor het roosteradvies -- eerste stap richting een echte app
(native/cross-platform), los van de Streamlit-dashboard. Ontsluit exact
dezelfde pijplijn als pages/1_Rooster.py (orders -> dagomzet -> weer ->
events -> features -> model -> roosteradvies), maar zonder Streamlit: een
toekomstige app-frontend (of iemand anders) kan hier gewoon tegenaan praten.

Bewust dezelfde bouwstenen als src/pipeline.py, maar zonder de
@st.cache_data-laag (die is Streamlit-specifiek en hoort niet in een
backend) -- geen van de "beschermde" bestanden (model.py/staffing.py/
purchasing.py/rooster.py) is aangepast.

Run: source .venv/bin/activate && uvicorn api:app --reload --port 8000
Test: curl -F "orders=@data/kassa_orderregels.csv" -F "locatie=Cafe Laurierboom, Amsterdam" \
        http://localhost:8000/rooster
"""
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.data_loading import load_kassa_orderregels as read_orders_csv
from src.aggregation import aggregate_orders_to_daily
from src.weather import geocode, fetch_historical_weather
from src.features import add_calendar_features, add_event_flags, add_weather_features, add_lag_features, add_rolling_features
from src.split import time_split
from src.pipeline import load_local_events
import model
import rooster

app = FastAPI(title="Vooruitzicht API", version="0.1.0")

# Tijdens ontwikkeling open voor alle origins -- een mobiele/native app draait
# niet op hetzelfde domein als deze API. Bij een echte release dichtzetten
# naar de daadwerkelijke app-origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bouw_features(orders: pd.DataFrame, locatie: str) -> pd.DataFrame:
    daily = aggregate_orders_to_daily(orders)

    loc = geocode(locatie)
    start = daily["datum"].min().strftime("%Y-%m-%d")
    end = daily["datum"].max().strftime("%Y-%m-%d")
    weather = fetch_historical_weather(loc["lat"], loc["lon"], start, end)

    df = daily.merge(weather, on="datum", how="left")
    df[["temp_c", "neerslag_mm", "zon_index"]] = df[["temp_c", "neerslag_mm", "zon_index"]].ffill()

    events = load_local_events()
    if events is not None:
        naam_per_datum = events.drop_duplicates("datum").set_index("datum")["naam"]
        df["event"] = df["datum"].map(naam_per_datum)
    else:
        df["event"] = None

    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target=model.TARGET)
    df = add_rolling_features(df, target=model.TARGET)
    return df


@app.get("/")
def root():
    return {"status": "ok", "service": "vooruitzicht-api"}


@app.post("/rooster")
def rooster_advies(
    orders: UploadFile = File(..., description="kassa-orderregels CSV, zelfde formaat als data/kassa_orderregels.csv"),
    locatie: str = Form("Cafe Laurierboom, Amsterdam"),
    norm_omzet_per_uur: float = Form(100.0),
):
    try:
        ruwe_orders = read_orders_csv(path=io.BytesIO(orders.file.read()))
    except Exception as e:
        raise HTTPException(400, f"kon het orderbestand niet lezen: {e}")

    try:
        df = _bouw_features(ruwe_orders, locatie)
    except Exception as e:
        raise HTTPException(400, f"kon weer/locatie niet ophalen: {e}")

    dates = np.sort(df["datum"].unique())
    if len(dates) < 30:
        raise HTTPException(400, "te weinig dagen in de upload om een testperiode van te maken")
    test_start = pd.Timestamp(dates[int(len(dates) * 0.8)]).strftime("%Y-%m-%d")
    train, test = time_split(df, test_start)
    if train.empty or test.empty:
        raise HTTPException(400, "te weinig dagen in de upload om een testperiode van te maken")

    aandeel = rooster.dagdeel_aandeel_per_weekday(ruwe_orders, test_start)
    aandeel_per_dag = aandeel.loc[test["datum"].dt.weekday].to_numpy()
    omzet_pred = model.predict(train, test, model.TARGET)
    advies = rooster.roosteradvies(
        omzet_pred, test["datum"].reset_index(drop=True), aandeel_per_dag, norm_omzet_per_uur
    )
    advies_records = advies.assign(datum=advies["datum"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")

    return {
        "test_periode": {
            "start": test_start,
            "eind": test["datum"].max().strftime("%Y-%m-%d"),
            "aantal_dagen": len(test),
        },
        "weer_samenvatting": {
            "gem_temp_c": round(float(test["temp_c"].mean()), 1),
            "regendagen": int((test["neerslag_mm"] > 1.0).sum()),
        },
        "dagdelen": rooster.DAGDELEN,
        "advies": advies_records,
    }
