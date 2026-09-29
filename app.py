"""Streamlit dashboard: upload a sensor CSV, then view assets, graphs, and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_preprocessing import (
    SENSOR_COLUMNS,
    add_rul_if_missing,
    apply_scaler,
    clean_sensor_frame,
    clip_rul,
    ensure_raw_data,
    fit_scaler,
    load_raw,
    split_by_unit,
)
from src.feature_engineering import add_health_index, add_rolling_features, last_cycle_per_unit
from src.anomaly_detection import anomaly_threshold, isolation_forest, score_isolation_forest
from src.rul_prediction import train_gbr

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Dashboard")
st.caption("Upload a sensor CSV to see fleet assets, anomaly graphs, and predicted remaining useful life.")

COLUMN_ALIASES = {
    "unit_id": ["unit_id", "asset_id", "engine_id", "machine_id"],
    "cycle": ["cycle", "time", "step", "operating_hours"],
    "temperature": ["temperature", "temp", "t1"],
    "vibration": ["vibration", "vib", "accel"],
    "pressure": ["pressure", "press"],
    "rpm": ["rpm", "speed", "rotational_speed"],
    "current": ["current", "amps", "amp"],
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower = {c.lower(): c for c in out.columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[lower[alias]] = canonical
                break
    return out.rename(columns=mapping)


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in ["unit_id", "cycle", *SENSOR_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + ". Need unit_id, cycle, temperature, vibration, pressure, rpm, current."
        )


@st.cache_data(show_spinner=False)
def score_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, float, bool]:
    raw_had_rul = "RUL" in df.columns or "rul" in [c.lower() for c in df.columns]
    df = _rename_columns(df)
    _validate(df)
    df = add_rul_if_missing(clean_sensor_frame(df))
    df = clip_rul(df)

    n_units = df["unit_id"].nunique()
    if n_units >= 4:
        train, _test = split_by_unit(df)
    else:
        train = df.copy()

    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    fleet = apply_scaler(df, scaler)
    fleet = add_health_index(add_rolling_features(fleet))
    train_f = add_health_index(add_rolling_features(train_s))

    skip = {"unit_id", "cycle", "RUL", "failure", "fail_mode"}
    feature_cols = [c for c in train_f.columns if c not in skip]
    for col in feature_cols:
        if col not in fleet.columns:
            fleet[col] = 0.0

    healthy = train_s[train_s["RUL"] > 40] if "RUL" in train_s.columns else train_s
    if len(healthy) < 32:
        healthy = train_s.iloc[: max(len(train_s) // 2, 16)]
    X_healthy = healthy[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    X_fleet = fleet[SENSOR_COLUMNS].to_numpy(dtype=np.float32)

    iforest = isolation_forest(X_healthy, contamination=0.08)
    gbr = train_gbr(
        train_f[feature_cols].to_numpy(dtype=np.float32),
        train_f["RUL"].to_numpy(dtype=np.float32),
    )

    healthy_scores = score_isolation_forest(iforest, X_healthy)
    thr = anomaly_threshold(healthy_scores, quantile=0.95)
    scores = score_isolation_forest(iforest, X_fleet)
    fleet["anomaly_score"] = scores
    fleet["is_anomaly"] = (scores > thr).astype(int)
    fleet["pred_RUL"] = np.clip(
        gbr.predict(fleet[feature_cols].to_numpy(dtype=np.float32)), 0, None
    )
    return fleet, float(thr), raw_had_rul


def render_dashboard(fleet: pd.DataFrame, thr: float, had_true_rul: bool) -> None:
    latest = last_cycle_per_unit(fleet).sort_values("pred_RUL")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", int(latest["unit_id"].nunique()))
    c2.metric("Anomalous now", int(latest["is_anomaly"].sum()))
    c3.metric("Median predicted RUL", f"{latest['pred_RUL'].median():.0f} cycles")
    c4.metric("Min predicted RUL", f"{latest['pred_RUL'].min():.0f} cycles")

    st.subheader("Predicted values by asset")
    display_cols = [
        "unit_id",
        "cycle",
        "health_index",
        "anomaly_score",
        "is_anomaly",
        "pred_RUL",
        *SENSOR_COLUMNS,
    ]
    if had_true_rul and "RUL" in latest.columns:
        display_cols.insert(6, "RUL")
    st.dataframe(latest[display_cols].round(3), use_container_width=True, hide_index=True)

    st.subheader("Graphs")
    units = sorted(fleet["unit_id"].unique())
    unit = st.selectbox("Inspect asset", units)
    unit_df = fleet[fleet["unit_id"] == unit].sort_values("cycle")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(unit_df["cycle"], unit_df["health_index"], label="Health index")
        ax.plot(unit_df["cycle"], unit_df["anomaly_score"], label="Anomaly score")
        ax.axhline(thr, color="red", ls="--", label="Anomaly threshold")
        ax.set_xlabel("Cycle")
        ax.set_title(f"Asset {unit} — degradation & anomalies")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with g2:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        if had_true_rul:
            ax.plot(unit_df["cycle"], unit_df["RUL"], label="True RUL")
        ax.plot(unit_df["cycle"], unit_df["pred_RUL"], label="Predicted RUL")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Cycles remaining")
        ax.set_title(f"Asset {unit} — remaining useful life")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    fig, axes = plt.subplots(len(SENSOR_COLUMNS), 1, figsize=(10, 10), sharex=True)
    for ax, col in zip(axes, SENSOR_COLUMNS):
        ax.plot(unit_df["cycle"], unit_df[col], color="steelblue")
        ax.set_ylabel(col)
    axes[-1].set_xlabel("Cycle")
    fig.suptitle(f"Asset {unit} — sensor traces (scaled)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("All scored rows")
    row_cols = ["unit_id", "cycle", "health_index", "anomaly_score", "is_anomaly", "pred_RUL"]
    if had_true_rul:
        row_cols.append("RUL")
    st.dataframe(fleet[row_cols].round(3), use_container_width=True, hide_index=True, height=280)


st.markdown(
    """
**Required columns:** `unit_id`, `cycle`, `temperature`, `vibration`, `pressure`, `rpm`, `current`.  
Optional: `RUL`. If RUL is omitted, it is estimated from each asset's last recorded cycle (run-to-failure assumption).
"""
)

left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
with right:
    st.write("")
    st.write("")
    use_sample = st.button("Use sample fleet", use_container_width=True)

source_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        source_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif use_sample:
    ensure_raw_data()
    source_df = load_raw()
    st.session_state["sample_loaded"] = True
elif st.session_state.get("sample_loaded"):
    source_df = load_raw()

if source_df is None:
    st.info("Upload a CSV (or click **Use sample fleet**) to generate assets, graphs, and predicted RUL.")
    st.stop()

st.success(f"Loaded {len(source_df):,} rows.")
st.dataframe(source_df.head(), use_container_width=True, hide_index=True)

try:
    with st.spinner("Training models and scoring your file…"):
        fleet, thr, had_rul = score_dataframe(source_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

render_dashboard(fleet, thr, had_rul)
