# Horeca drukte- en inkoopvoorspelling (prototype)

## Setup

```bash
cd /Users/julianbaks/claude/horeca
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Losse scripts (command line)

```bash
python run_baselines.py   # sanity check: 4 methodes tegen de referentie-MAPE's
python model.py           # omzetmodel (HistGradientBoostingRegressor) + scoreboard
python staffing.py        # voorspelde omzet -> aanbevolen uren -> euro's
python purchasing.py      # voorspeld verbruik per product -> besteladvies
```

## Dashboard

```bash
streamlit run app.py
```

Opent op `http://localhost:8501`. Bovenaan kun je een eigen `dagstaat.csv` en
`kassa_orderregels.csv` uploaden; zonder upload gebruikt het dashboard de
synthetische bestanden uit `data/`. Let op: de pipeline rekent op
`dagstaat.csv` (en `verbruik_theoretisch.csv` voor inkoop) -- een geuploade
`kassa_orderregels.csv` wordt ingelezen en getoond, maar voedt de modellen
op dit moment niet rechtstreeks.
