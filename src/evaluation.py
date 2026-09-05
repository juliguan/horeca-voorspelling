"""Evaluatie-metrics en een simpel scoreboard om modellen tegen elkaar
(en tegen de bekende baselines) af te zetten."""
import numpy as np
import pandas as pd


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, in procenten. Gaat ervan uit dat y_true
    geen nullen bevat (in de test-periode hier is dat het geval; check dit als
    je een andere periode of ander target gebruikt)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if (y_true == 0).any():
        raise ValueError(
            "y_true bevat nullen -- MAPE is dan ongedefinieerd. "
            "Filter deze dagen eruit of gebruik een andere metric."
        )
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score(y_true, y_pred) -> dict:
    return {"mape": mape(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def scoreboard(results: dict) -> pd.DataFrame:
    """results: {naam: (y_true, y_pred)} -> nette tabel gesorteerd op MAPE."""
    rows = []
    for name, (y_true, y_pred) in results.items():
        row = {"model": name}
        row.update(score(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
