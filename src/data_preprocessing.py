"""Load, clean, and scale multivariate sensor time series for PdM."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "current",
]
ID_COLUMNS = ["unit_id", "cycle"]
TARGET_COLUMNS = ["RUL", "failure"]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def generate_synthetic_fleet(
    n_units: int = 40,
    min_cycles: int = 80,
    max_cycles: int = 220,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a fleet of degrading assets with sensor drift and late-life faults."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for unit_id in range(1, n_units + 1):
        life = int(rng.integers(min_cycles, max_cycles + 1))
        fail_mode = rng.choice(["bearing", "thermal", "pressure"], p=[0.45, 0.35, 0.20])

        temp0 = rng.normal(72, 3)
        vib0 = rng.normal(0.35, 0.04)
        press0 = rng.normal(101, 2)
        rpm0 = rng.normal(1480, 25)
        curr0 = rng.normal(12.5, 0.4)

        for cycle in range(1, life + 1):
            progress = cycle / life
            wear = progress ** 1.7

            temperature = temp0 + 18 * wear + rng.normal(0, 0.6)
            vibration = vib0 + 1.4 * wear + rng.normal(0, 0.03)
            pressure = press0 - 12 * wear + rng.normal(0, 0.5)
            rpm = rpm0 - 80 * wear + rng.normal(0, 8)
            current = curr0 + 4.5 * wear + rng.normal(0, 0.12)

            if fail_mode == "bearing":
                vibration += 0.8 * wear**2
            elif fail_mode == "thermal":
                temperature += 10 * wear**2
                current += 1.2 * wear
            else:
                pressure -= 8 * wear**2

            # Occasional spikes near end of life
            if progress > 0.85 and rng.random() < 0.08:
                vibration += rng.uniform(0.4, 1.1)
                temperature += rng.uniform(2, 6)

            failure = int(cycle == life)
            rows.append(
                {
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "temperature": temperature,
                    "vibration": vibration,
                    "pressure": pressure,
                    "rpm": rpm,
                    "current": current,
                    "fail_mode": fail_mode,
                    "RUL": life - cycle,
                    "failure": failure,
                }
            )

    return pd.DataFrame(rows)


def ensure_raw_data(path: Path | None = None, **gen_kwargs) -> Path:
    """Write synthetic CSV if no raw dataset exists yet."""
    path = path or (RAW_DIR / "fleet_sensors.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        generate_synthetic_fleet(**gen_kwargs).to_csv(path, index=False)
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = ensure_raw_data(path)
    df = pd.read_csv(path)
    required = ID_COLUMNS + SENSOR_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data missing columns: {missing}")
    return df


def clean_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(ID_COLUMNS).reset_index(drop=True)
    out[SENSOR_COLUMNS] = out[SENSOR_COLUMNS].interpolate(limit_direction="both")
    out = out.dropna(subset=SENSOR_COLUMNS)
    out = out[out["cycle"] > 0]
    return out


def add_rul_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "RUL" not in out.columns:
        max_cycle = out.groupby("unit_id")["cycle"].transform("max")
        out["RUL"] = max_cycle - out["cycle"]
    if "failure" not in out.columns:
        out["failure"] = (out["RUL"] == 0).astype(int)
    return out


def clip_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Cap early-life RUL so the model focuses on the degradation window."""
    out = df.copy()
    out["RUL"] = out["RUL"].clip(upper=max_rul)
    return out


def split_by_unit(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = df["unit_id"].unique()
    train_units, test_units = train_test_split(
        units, test_size=test_size, random_state=seed
    )
    train = df[df["unit_id"].isin(train_units)].copy()
    test = df[df["unit_id"].isin(test_units)].copy()
    return train, test


def fit_scaler(train: pd.DataFrame, columns: Iterable[str] | None = None) -> MinMaxScaler:
    columns = list(columns or SENSOR_COLUMNS)
    scaler = MinMaxScaler()
    scaler.fit(train[columns])
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: MinMaxScaler, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    columns = list(columns or SENSOR_COLUMNS)
    out = df.copy()
    out[columns] = scaler.transform(out[columns])
    return out


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_csv(path, index=False)
    return path


def prepare_datasets(
    raw_path: Path | None = None, max_rul: int = 125
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    df = add_rul_if_missing(clean_sensor_frame(load_raw(raw_path)))
    df = clip_rul(df, max_rul=max_rul)
    train, test = split_by_unit(df)
    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    test_s = apply_scaler(test, scaler)
    save_processed(train_s, "train.csv")
    save_processed(test_s, "test.csv")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")
    return train_s, test_s, scaler
