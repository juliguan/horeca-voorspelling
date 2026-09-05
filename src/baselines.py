"""
Referentie-baselines uit de opdracht, als sanity check op de schil (inlezen,
features, split, evaluatie). Dit zijn GEEN suggesties voor het uiteindelijke
model -- puur de drie bekende vergelijkingspunten:

    weekdaggemiddelde                          ~27% MAPE
    zelfde dag 4 weken terug                   ~22% MAPE
    weekdag + maand + weer + event + trend     ~12% MAPE (simpele lineaire regressie)

Als deze functies (op deze data, met deze split) in de buurt van die getallen
uitkomen, weet je dat data_loading/features/split/evaluation kloppen en kun je
vertrouwen op het scoreboord voor je eigen model.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def predict_weekday_average(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Gemiddelde van 'target' per weekdag, berekend op train, toegepast op test."""
    avg_per_weekday = train.groupby(train["datum"].dt.weekday)[target].mean()
    return test["datum"].dt.weekday.map(avg_per_weekday).to_numpy()


def predict_same_weekday_n_weeks_ago(full_df: pd.DataFrame, test_dates: pd.Series, target: str, n_weeks: int = 4) -> np.ndarray:
    """Werkelijke waarde van dezelfde weekdag n_weeks geleden. Gebruikt full_df
    (train+test aan elkaar, chronologisch) omdat dit alleen op het verleden
    t.o.v. elke voorspelde dag leunt -- geen toekomstlekkage.

    Lookup gaat op datum (niet op rij-positie), zodat dit ook werkt als
    train/test een reset index hebben (zoals na time_split)."""
    by_date = full_df.set_index("datum")[target]
    lookup_dates = test_dates - pd.Timedelta(days=7 * n_weeks)
    return by_date.reindex(lookup_dates).to_numpy()


def _linear_feature_pipeline(categorical_cols, numeric_cols) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("lr", LinearRegression())])


def predict_simple_linear(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """weekday + maand + weer (temp/neerslag/zon) + event + trend, gewone OLS.

    Extra: interactieterm weekend (za/zo) x zon_index, als losse kolom naast
    de bestaande features."""
    categorical_cols = ["weekday_num", "month", "event"]
    numeric_cols = ["temp_c", "neerslag_mm", "zon_index", "trend", "weekend_x_zon"]
    cols = categorical_cols + numeric_cols

    train_f = train.copy()
    test_f = test.copy()
    train_f["event"] = train_f["event"].fillna("geen").replace("", "geen")
    test_f["event"] = test_f["event"].fillna("geen").replace("", "geen")
    for df_f in (train_f, test_f):
        is_weekend = df_f["weekday_num"].isin([5, 6]).astype(int)
        df_f["weekend_x_zon"] = is_weekend * df_f["zon_index"]

    pipe = _linear_feature_pipeline(categorical_cols, numeric_cols)
    pipe.fit(train_f[cols], train_f[target])
    return pipe.predict(test_f[cols])
