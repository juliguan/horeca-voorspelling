"""
Feature-bouwstenen op dagniveau (1 rij per datum, zoals dagstaat.csv).

Dit is bewust een gereedschapskist van losse functies, geen "build_features()"
die al kiest wat er in het model gaat. Welke features je gebruikt en hoe
(bijv. wel/geen lags, wel/geen one-hot) is een modelkeuze -- die maak je zelf
in je eigen model script. Hier zit alleen de verwerking die voor elk model
hetzelfde is (kalender, weer doorgeven, event-encoding, lags/rolling ophalen).

Alle functies zijn puur: ze nemen een dagstaat-achtige DataFrame (met kolom
'datum', oplopend gesorteerd, geen gaten) en geven diezelfde df terug met
extra kolommen.
"""
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """weekday (0=maandag), maand, dag-van-jaar, en trend (dagen sinds eerste datum)."""
    df = df.copy()
    df["weekday_num"] = df["datum"].dt.weekday
    df["month"] = df["datum"].dt.month
    df["day_of_year"] = df["datum"].dt.dayofyear
    df["trend"] = (df["datum"] - df["datum"].min()).dt.days
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot van de 'event' kolom (NaN/lege string wordt 'geen') + simpele is_event flag."""
    df = df.copy()
    event = df["event"].fillna("").replace("", "geen")
    df["is_event"] = (event != "geen").astype(int)
    dummies = pd.get_dummies(event, prefix="event", dtype=int)
    return pd.concat([df, dummies], axis=1)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weer staat al in dagstaat (temp_c, neerslag_mm, zon_index) -- dit voegt alleen
    afgeleiden toe die vaak handig zijn (droog/nat, koud/warm bucket)."""
    df = df.copy()
    df["is_droog"] = (df["neerslag_mm"] == 0).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags_in_days: tuple = (7, 14, 21, 28),
) -> pd.DataFrame:
    """Voegt target-waarde van N dagen terug toe (bijv. lag_7 = zelfde weekdag vorige week).

    LET OP: dit gebruikt alleen verleden t.o.v. elke rij (shift), dus veilig voor
    een chronologische train/test split -- geen lekkage uit de toekomst.
    """
    df = df.copy()
    for lag in lags_in_days:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows: tuple = (7, 28),
) -> pd.DataFrame:
    """Voortschrijdend gemiddelde/std van het verleden (shift(1) eerst, dus de waarde
    van vandaag zelf lekt niet mee)."""
    df = df.copy()
    shifted = df[target].shift(1)
    for w in windows:
        df[f"{target}_rollmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{target}_rollstd_{w}"] = shifted.rolling(w, min_periods=1).std()
    return df


def add_same_weekday_n_weeks_ago(df: pd.DataFrame, target: str, n_weeks: int = 4) -> pd.DataFrame:
    """Waarde van dezelfde weekdag N weken geleden (bouwsteen voor de 'zelfde dag
    4 weken terug'-baseline, en een prima los feature)."""
    df = df.copy()
    df[f"{target}_same_weekday_{n_weeks}w_ago"] = df[target].shift(7 * n_weeks)
    return df
