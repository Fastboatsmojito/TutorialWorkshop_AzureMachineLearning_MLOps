"""Scoring script for the AESO price-forecast managed online endpoint.

Request body (all fields optional except nothing is strictly required):
{
  "start": "2026-01-15T00:00:00",   # ISO; default = next top of hour (UTC)
  "horizon_hours": 168,              # default 24, max 720
  "demand_multiplier": 1.10,         # scenario: scale load (e.g. +10%)
  "temperature_offset_c": -5.0,      # scenario: shift temperature
  "exogenous": [                     # optional explicit drivers per hour
    {"timestamp": "...", "ail_demand_mw": 9800, "temperature_c": -12,
     "wind_generation_mw": 400, "gas_price_aeco": 2.7}
  ]
}

Response:
{
  "units": "CAD/MWh",
  "model": "aeso-price-forecaster",
  "forecast": [{"timestamp": "...", "predicted_price": 73.4,
                "ail_demand_mw": ..., "temperature_c": ..., ...}],
  "summary": {"avg": .., "min": .., "max": .., "peak_hour": ".."}
}
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# These helpers are bundled into the model folder at deploy time (see
# deploy_endpoint.py), so `common` is importable from the model dir.
import sys

_MODEL = None
_CLIM = None
_CLIM_GLOBAL = None
_CARD = {}
_EXOG = ["ail_demand_mw", "temperature_c", "wind_generation_mw", "gas_price_aeco"]


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["dayofyear"] = ts.dt.dayofyear
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["is_holiday"] = 0
    out["net_load_mw"] = out["ail_demand_mw"] - out["wind_generation_mw"]
    return out


_FEATURES = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "ail_demand_mw", "temperature_c", "wind_generation_mw", "gas_price_aeco",
    "net_load_mw",
]


def init() -> None:
    global _MODEL, _CLIM, _CLIM_GLOBAL, _CARD
    model_root = Path(os.getenv("AZUREML_MODEL_DIR", "."))
    # Anchor on climatology.csv so we pick OUR artifact folder, not the
    # MLflow-flavored copy that also contains a model.pkl.
    clim_candidates = list(model_root.rglob("climatology.csv"))
    if not clim_candidates:
        raise FileNotFoundError(f"climatology.csv not found under {model_root}")
    base = clim_candidates[0].parent
    _MODEL = joblib.load(base / "model.pkl")
    _CLIM = pd.read_csv(base / "climatology.csv")
    _CLIM_GLOBAL = _CLIM[_EXOG].mean().to_dict()
    card_path = base / "model_card.json"
    if card_path.exists():
        _CARD = json.loads(card_path.read_text())
    print("Model loaded from", base)


def _fill_from_climatology(months: np.ndarray, hours: np.ndarray) -> pd.DataFrame:
    key = _CLIM.set_index(["month", "hour"])
    rows = []
    for m, h in zip(months, hours):
        try:
            rows.append(key.loc[(m, h)][_EXOG].to_dict())
        except KeyError:
            rows.append(dict(_CLIM_GLOBAL))
    return pd.DataFrame(rows)


def run(raw_data):
    body = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
    body = body or {}

    horizon = int(body.get("horizon_hours", 24))
    horizon = max(1, min(horizon, 720))

    if body.get("start"):
        start = pd.to_datetime(body["start"])
    else:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = pd.Timestamp(now + timedelta(hours=1)).tz_localize(None)

    timestamps = pd.date_range(start=start, periods=horizon, freq="h")

    if body.get("exogenous"):
        exo = pd.DataFrame(body["exogenous"])
        exo["timestamp"] = pd.to_datetime(exo["timestamp"])
        frame = exo
    else:
        months = timestamps.month.to_numpy()
        hours = timestamps.hour.to_numpy()
        exo = _fill_from_climatology(months, hours)
        exo.insert(0, "timestamp", timestamps)
        frame = exo

    # Scenario knobs.
    demand_mult = float(body.get("demand_multiplier", 1.0))
    temp_offset = float(body.get("temperature_offset_c", 0.0))
    frame["ail_demand_mw"] = frame["ail_demand_mw"] * demand_mult
    frame["temperature_c"] = frame["temperature_c"] + temp_offset

    feat = _add_calendar_features(frame)
    preds = _MODEL.predict(feat[_FEATURES])
    preds = np.clip(preds, 0, 999.99).round(2)

    forecast = []
    for i, ts in enumerate(frame["timestamp"]):
        forecast.append(
            {
                "timestamp": pd.Timestamp(ts).isoformat(),
                "predicted_price": float(preds[i]),
                "ail_demand_mw": round(float(frame["ail_demand_mw"].iloc[i]), 1),
                "temperature_c": round(float(frame["temperature_c"].iloc[i]), 1),
                "wind_generation_mw": round(float(frame["wind_generation_mw"].iloc[i]), 1),
                "gas_price_aeco": round(float(frame["gas_price_aeco"].iloc[i]), 3),
            }
        )

    peak_i = int(np.argmax(preds))
    summary = {
        "avg": round(float(np.mean(preds)), 2),
        "min": round(float(np.min(preds)), 2),
        "max": round(float(np.max(preds)), 2),
        "peak_hour": pd.Timestamp(frame["timestamp"].iloc[peak_i]).isoformat(),
        "horizon_hours": horizon,
        "scenario": {"demand_multiplier": demand_mult, "temperature_offset_c": temp_offset},
    }

    return {
        "units": "CAD/MWh",
        "model": _CARD.get("name", "aeso-price-forecaster"),
        "forecast": forecast,
        "summary": summary,
    }
