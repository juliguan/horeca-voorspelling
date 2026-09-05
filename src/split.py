"""Train/test split op datum -- nooit random, want dit is een tijdreeks."""
import pandas as pd

TEST_START = "2026-03-01"  # zoals afgesproken: test = vanaf maart 2026


def time_split(df: pd.DataFrame, test_start: str = TEST_START, date_col: str = "datum"):
    """Geeft (train, test) terug. Alles voor test_start is train, de rest is test.

    Als je lag/rolling features hebt toegevoegd op de volledige df (features.py
    kijkt alleen naar het verleden per rij), is dit achteraf splitsen veilig --
    er lekt geen informatie uit de toekomst in de train-set.
    """
    train = df[df[date_col] < test_start].reset_index(drop=True)
    test = df[df[date_col] >= test_start].reset_index(drop=True)
    return train, test
