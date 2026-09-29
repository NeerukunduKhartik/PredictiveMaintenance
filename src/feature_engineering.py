"""Rolling statistics and sequence windows for anomaly and RUL models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_preprocessing import SENSOR_COLUMNS


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 15),
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    sensors = sensors or SENSOR_COLUMNS
    parts = [df]
    grouped = df.groupby("unit_id", group_keys=False)

    for w in windows:
        rolled = grouped[sensors].transform(
            lambda g: g.rolling(window=w, min_periods=1).mean()
        )
        rolled = rolled.add_suffix(f"_rmean_{w}")
        std = grouped[sensors].transform(
            lambda g: g.rolling(window=w, min_periods=1).std().fillna(0)
        )
        std = std.add_suffix(f"_rstd_{w}")
        parts.extend([rolled, std])

    out = pd.concat(parts, axis=1)
    return out


def health_index(df: pd.DataFrame, sensors: list[str] | None = None) -> pd.Series:
    """Simple composite: mean of inverted 'healthy' sensors after scaling."""
    sensors = sensors or SENSOR_COLUMNS
    # After MinMax scaling, higher temp/vib/current and lower pressure/rpm indicate wear.
    wear = (
        df["temperature"]
        + df["vibration"]
        + df["current"]
        + (1 - df["pressure"])
        + (1 - df["rpm"])
    ) / 5.0
    return wear.rename("health_index")


def add_health_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["health_index"] = health_index(out)
    return out


def sequence_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "RUL",
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_windows, window, n_features) tensors, one step per cycle."""
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for _, group in df.groupby("unit_id"):
        group = group.sort_values("cycle")
        values = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        if len(group) < window:
            continue
        for i in range(window - 1, len(group)):
            X_list.append(values[i - window + 1 : i + 1])
            y_list.append(targets[i])

    if not X_list:
        return np.empty((0, window, len(feature_cols))), np.empty((0,))
    return np.stack(X_list), np.asarray(y_list, dtype=np.float32)


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("cycle").groupby("unit_id", as_index=False).tail(1)
