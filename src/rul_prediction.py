"""Remaining useful life: Gradient Boosting baseline and LSTM sequence model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "rul"


def _keras():
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras, layers


def train_gbr(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 250,
    max_depth: int = 3,
    seed: int = 42,
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def build_lstm(window: int, n_features: int) -> Any:
    keras, layers = _keras()
    model = keras.Sequential(
        [
            layers.Input(shape=(window, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ],
        name="rul_lstm",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
) -> Any:
    keras, _layers = _keras()
    model = build_lstm(X.shape[1], X.shape[2])
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early],
        verbose=0,
    )
    return model


def save_gbr(model: GradientBoostingRegressor, name: str = "gbr_rul.joblib") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path


def load_gbr(name: str = "gbr_rul.joblib") -> GradientBoostingRegressor:
    return joblib.load(MODEL_DIR / name)


def save_lstm(model: Any, name: str = "lstm_rul.keras") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    model.save(path)
    return path


def load_lstm(name: str = "lstm_rul.keras") -> Any:
    keras, _layers = _keras()
    return keras.models.load_model(MODEL_DIR / name)
