# Singapore Flood Risk Map

Interactive Streamlit project to visualize Singapore weather and flood signals by subzone and estimate short-term flood risk with a transparent rule-based proxy while labeled flood outcomes are still limited.

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://singapore-flood-risk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](#quick-start)

## Live App
- Streamlit: **https://singapore-flood-risk.streamlit.app/**

## App Preview

![Singapore Flood Risk App](assets/app-preview.png)

## What This Project Does
- Displays Singapore subzones on an interactive map.
- Lets you inspect **single features** one by one (rainfall, humidity, lightning, forecast, etc.).
- Shows a **combined flood-risk layer** computed from a configurable proxy formula.
- Supports:
  - **Historical Replay** (timestamp slider)
  - **Live API Snapshot** (real-time pull from data.gov.sg)
- Provides a dashboard with metrics, charts, and ranked risk tables.

## Why A Proxy Model
Flood labels are currently limited/non-annotated for supervised training.  
So the current prediction is a transparent formula combining last-1h weather context:
- rainfall accumulation (15/60 min)
- humidity
- lightning activity
- 2-hour forecast signal
- ongoing flood-alert pressure

This design is intentional so the UI/data pipeline can be production-ready while a true ML model is prepared.

## Architecture
1. **Ingestion**
- Pull data from data.gov.sg weather/flood APIs.
- Build timestamped subzone features (`subzone + timestamp_5min`).

2. **Feature Layer**
- Weather + forecast + event indicators normalized per timestamp.

3. **Risk Layer**
- Rule-based risk scoring (`0..1`) with configurable weights.

4. **Visualization**
- Streamlit + PyDeck choropleth map.
- Metrics/charts/tables under the map.

## Repository Status
- The repository includes a committed **boundary GeoJSON** and a **processed demo snapshot** so the app runs out of the box.
- Secrets and raw exports are still ignored.
- The canonical processed-feature builder is `scripts/build_dataset_from_datagov.py` or, if raw CSVs already exist locally, `scripts/build_processed_features_fast.py`.

## Quick Start
```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# Optional: install project metadata + dev tooling
pip install -e .[dev]

# 3) Add your API key
# .env -> api-key=...

# 4) Run app
streamlit run app.py
```

## Data Pipeline
```bash
# Download raw weather/flood exports and rebuild the processed feature table
python scripts/build_dataset_from_datagov.py \
  --start-date 2026-03-01 \
  --end-date 2026-03-05

# Rebuild the processed feature table only, from existing raw CSV files
python scripts/build_processed_features_fast.py \
  --raw-dir data/raw \
  --output data/processed/subzone_weather_features.csv
```

## Tests
```bash
ruff check .
python -m unittest discover -s tests -v
```

## Data Policy
This repository ignores secrets and raw/large data artifacts by default.
Ignored by default:
- `data/raw/`
- `data/raw_v2/`
- `data/processed/`
- `data/*.csv`
- `data/*.geojson`
- `data/*.json`
- `data/*.parquet`
- `.env`

## Repository Structure
```text
.
├── app.py
├── scripts/
│   ├── build_dataset_from_datagov.py
│   ├── build_processed_features_fast.py
│   └── build_rainfall_history.py
├── floodlib/
│   ├── common.py
│   ├── feature_pipeline.py
│   └── risk_model.py
├── tests/
│   ├── test_feature_pipeline.py
│   └── test_risk_model.py
├── data/
│   ├── MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson
│   └── processed/subzone_weather_features.csv
└── assets/
    ├── .gitkeep
    └── app-preview.png   # add manually
```

## Next Milestones
- Train supervised flood model once sufficient labeled events are available.
- Add calibration + backtesting dashboard.
- Add uncertainty band per subzone risk score.

---
Maintained by [juliennigou](https://github.com/juliennigou)
