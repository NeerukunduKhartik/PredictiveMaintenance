# Predictive Maintenance

Intern project for **anomaly detection** and **remaining useful life (RUL)** on multivariate industrial sensor data.

If no dataset is present under `data/raw/`, the pipeline generates a synthetic fleet (`fleet_sensors.csv`) with temperature, vibration, pressure, RPM, and current drifting toward failure.

## Layout

```
Predictive-Maintenance/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_rul_prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py
│   ├── rul_prediction.py
│   └── evaluation.py
├── models/
│   ├── autoencoder/
│   └── rul/
├── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks from the **project root** so `src` imports resolve:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

## Dashboard

```bash
streamlit run app.py
```

First launch trains a reconstruction autoencoder (healthy cycles) and a gradient-boosting RUL model, then caches them under `models/`.

## Pipeline

1. **Exploration** — fleet size, sensor drift, RUL histograms.
2. **Preprocessing** — unit-level train/test split, MinMax scaling, RUL cap.
3. **Anomaly detection** — Isolation Forest + autoencoder reconstruction error.
4. **RUL** — Gradient Boosting on rolling features; optional LSTM on cycle windows.

## Swap in real data

Place a CSV at `data/raw/fleet_sensors.csv` with at least:

`unit_id, cycle, temperature, vibration, pressure, rpm, current`

Optional: `RUL`, `failure`. If RUL is omitted it is derived as `max(cycle) - cycle` per unit.
