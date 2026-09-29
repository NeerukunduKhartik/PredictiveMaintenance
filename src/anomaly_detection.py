"""Isolation Forest baseline plus reconstruction autoencoder for sensor anomalies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data_preprocessing import SENSOR_COLUMNS

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "autoencoder"


def _keras():
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras, layers


def isolation_forest(
    X: np.ndarray, contamination: float = 0.08, seed: int = 42
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.decision_function(X)


def build_autoencoder(n_features: int, encoding_dim: int = 8) -> Any:
    keras, layers = _keras()
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    encoded = layers.Dense(encoding_dim, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(encoded)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_features, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="sensor_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    encoding_dim: int = 8,
) -> Any:
    keras, _layers = _keras()
    model = build_autoencoder(X_train.shape[1], encoding_dim=encoding_dim)
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early],
        verbose=0,
    )
    return model


def reconstruction_error(model: Any, X: np.ndarray) -> np.ndarray:
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def anomaly_threshold(errors: np.ndarray, quantile: float = 0.95) -> float:
    return float(np.quantile(errors, quantile))


def annotate_anomalies(
    df: pd.DataFrame,
    errors: np.ndarray,
    threshold: float,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    feature_cols = feature_cols or SENSOR_COLUMNS
    out = df.copy()
    out["recon_error"] = errors
    out["is_anomaly"] = (errors > threshold).astype(int)
    return out


def save_autoencoder(model: Any, name: str = "autoencoder.keras") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    model.save(path)
    return path


def load_autoencoder(name: str = "autoencoder.keras") -> Any:
    keras, _layers = _keras()
    return keras.models.load_model(MODEL_DIR / name)


def save_iforest(model: IsolationForest, name: str = "iforest.joblib") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path
