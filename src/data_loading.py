"""Inlezen van de ruwe CSV's uit data/. Puur IO, geen features/logica."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dagstaat(path: Path = DATA_DIR / "dagstaat.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datum"])
    return df.sort_values("datum").reset_index(drop=True)


def load_kassa_orderregels(path: Path = DATA_DIR / "kassa_orderregels.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["tijdstip"])
    df["datum"] = df["tijdstip"].dt.normalize()
    return df


def load_producten(path: Path = DATA_DIR / "producten.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_receptuur(path: Path = DATA_DIR / "receptuur.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_verbruik_theoretisch(path: Path = DATA_DIR / "verbruik_theoretisch.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datum"])
    return df.sort_values(["productcode", "datum"]).reset_index(drop=True)


def load_inkoop_historie(path: Path = DATA_DIR / "inkoop_historie.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["leverdatum"])
    return df.sort_values(["productcode", "leverdatum"]).reset_index(drop=True)


def load_derving_werkelijk(path: Path = DATA_DIR / "derving_werkelijk.csv") -> pd.DataFrame:
    """Ground truth bederf — alleen voor achteraf evalueren, niet als feature gebruiken."""
    df = pd.read_csv(path, parse_dates=["datum"])
    return df.sort_values(["productcode", "datum"]).reset_index(drop=True)
