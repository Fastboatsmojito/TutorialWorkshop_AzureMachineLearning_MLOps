"""Shared feature engineering used by BOTH training and online scoring.

Keeping this in one place guarantees the features the model is trained on are
identical to the features computed at inference time (a common MLOps failure
point known as training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays

    _AB_HOLIDAYS = _holidays.CountryHoliday("CA", subdiv="AB")
except Exception:  # pragma: no cover - holidays optional at serving time
    _AB_HOLIDAYS = None

# Exogenous drivers the model consumes (besides calendar features).
EXOG_COLUMNS = [
    "ail_demand_mw",
    "temperature_c",
    "wind_generation_mw",
    "gas_price_aeco",
]

TARGET_COLUMN = "pool_price"
TIME_COLUMN = "timestamp"


def add_calendar_features(df: pd.DataFrame, time_col: str = TIME_COLUMN) -> pd.DataFrame:
    """Add deterministic calendar features derivable from the timestamp alone."""
    out = df.copy()
    ts = pd.to_datetime(out[time_col])

    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["dayofyear"] = ts.dt.dayofyear
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encodings so the model understands 23:00 is close to 00:00.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    if _AB_HOLIDAYS is not None:
        out["is_holiday"] = ts.dt.date.map(lambda d: int(d in _AB_HOLIDAYS))
    else:
        out["is_holiday"] = 0

    # Net load: the single biggest driver of price. Demand the grid must serve
    # from dispatchable units after subtracting (near-zero-cost) wind.
    if {"ail_demand_mw", "wind_generation_mw"}.issubset(out.columns):
        out["net_load_mw"] = out["ail_demand_mw"] - out["wind_generation_mw"]

    return out


def feature_columns() -> list[str]:
    """Ordered list of columns the model is trained/served on."""
    calendar = [
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    return calendar + EXOG_COLUMNS + ["net_load_mw"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full transform: raw rows (timestamp + exogenous) -> model feature matrix."""
    enriched = add_calendar_features(df)
    return enriched[feature_columns()]
